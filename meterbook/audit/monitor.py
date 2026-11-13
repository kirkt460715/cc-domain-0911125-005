import datetime
import json
import math
import re
from collections import defaultdict
from decimal import Decimal
from meterbook.values import key, instant, encoded, canonical, digest, reject, MeterError
from meterbook.events import reconcile
from meterbook.contracts import registry
from meterbook.contracts.calendar import Calendar

DDL = '''
CREATE TABLE IF NOT EXISTS monitor_models(model_id TEXT PRIMARY KEY,contract_id TEXT NOT NULL,version INTEGER NOT NULL,payload TEXT NOT NULL,digest TEXT NOT NULL,created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS monitor_lineage(model_id TEXT NOT NULL,day TEXT NOT NULL,meter TEXT NOT NULL,quantity TEXT NOT NULL,evidence TEXT NOT NULL,PRIMARY KEY(model_id,day,meter));
'''

MIN_TRAIN_DAYS = 8
MIN_CALIBRATION_DAYS = 5
MAX_RANGE_DAYS = 400
EIGENVALUE_TOLERANCE = 1e-12

INTERPRETATIONS = {
    'none': 'in_control',
    't2': 'magnitude_shift_inside_learned_relation',
    'spe': 'learned_meter_relation_broken',
    'joint': 'magnitude_and_relation_shift',
}

ALERT_SEMANTICS = {
    't2': 'unusual overall magnitude inside the learned normal-relation subspace',
    'spe': 'learned relation between meters broken, visible in the residual space',
    'joint': 'both statistics exceed their frozen calibration quantiles; treat as a combined manual-review hint',
    'note': 'thresholds are empirical calibration quantiles frozen at fit time; exceedance rates are calibration observations, not guaranteed long-run error rates; alerts are review hints and never billing judgements',
}


def _timezone(value):
    if not isinstance(value, str):
        reject('monitor_timezone', 'Timezone must be UTC or a fixed offset')
    if value == 'UTC':
        return datetime.timezone.utc
    match = re.fullmatch(r'(?:UTC)?([+-])(\d{2}):(\d{2})', value)
    if not match:
        reject('monitor_timezone', 'Timezone must be UTC or a fixed offset like UTC+08:00')
    sign, hours, minutes = match.groups()
    offset = int(hours) * 60 + int(minutes)
    if offset >= 24 * 60:
        reject('monitor_timezone', 'Timezone offset must stay below 24 hours')
    delta = datetime.timedelta(minutes=offset)
    return datetime.timezone(delta if sign == '+' else -delta)


def _date(value, field):
    if not isinstance(value, str):
        reject('monitor_day', 'Day must use YYYY-MM-DD', field=field)
    try:
        parsed = datetime.date.fromisoformat(value)
    except ValueError as exc:
        raise MeterError('monitor_day', 'Day must use YYYY-MM-DD', field=field) from exc
    if not 2000 <= parsed.year <= 2200:
        reject('monitor_day', 'Unsupported business day', field=field)
    return parsed


def _parse_instant(value):
    return datetime.datetime.fromisoformat(instant(value).replace('Z', '+00:00'))


def _local_day(at, tzinfo):
    moment = datetime.datetime.fromisoformat(at.replace('Z', '+00:00'))
    return moment.astimezone(tzinfo).date().isoformat()


def _day_bounds(day, tzinfo):
    start = datetime.datetime.combine(day, datetime.time.min, tzinfo)
    end = datetime.datetime.combine(day + datetime.timedelta(days=1), datetime.time.min, tzinfo)
    return start.astimezone(datetime.timezone.utc), end.astimezone(datetime.timezone.utc)


def _months(first, last):
    year, month = first.year, first.month
    while (year, month) <= (last.year, last.month):
        yield f'{year:04}-{month:02}'
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)


def _fraction(value, code, minimum, maximum):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not minimum <= value <= maximum:
        reject(code, 'Expected a fraction within bounds', minimum=minimum, maximum=maximum)
    return float(value)


def _fit_scaling(rows):
    size = len(rows)
    width = len(rows[0])
    scaling = []
    for column in range(width):
        mean = sum(row[column] for row in rows) / size
        variance = sum((row[column] - mean) ** 2 for row in rows) / (size - 1)
        std = math.sqrt(variance)
        scaling.append({'mean': mean, 'std': std, 'scale': std if std > 0 else 1.0, 'constant': std == 0.0})
    return scaling


def _standardize(row, scaling):
    return [(row[column] - scaling[column]['mean']) / scaling[column]['scale'] for column in range(len(row))]


def _covariance(rows):
    size = len(rows)
    width = len(rows[0])
    means = [sum(row[column] for row in rows) / size for column in range(width)]
    return [[sum((row[i] - means[i]) * (row[j] - means[j]) for row in rows) / (size - 1) for j in range(width)] for i in range(width)]


def _eigh(matrix):
    size = len(matrix)
    a = [list(row) for row in matrix]
    vectors = [[1.0 if i == j else 0.0 for j in range(size)] for i in range(size)]
    for _ in range(200):
        pivot = max((abs(a[i][j]) for i in range(size) for j in range(i + 1, size)), default=0.0)
        scale = max(abs(a[i][i]) for i in range(size))
        if pivot <= 1e-13 * (1.0 + scale):
            break
        for i in range(size):
            for j in range(i + 1, size):
                if a[i][j] == 0.0:
                    continue
                theta = (a[j][j] - a[i][i]) / (2.0 * a[i][j])
                tangent = (1.0 if theta >= 0 else -1.0) / (abs(theta) + math.hypot(theta, 1.0))
                cosine = 1.0 / math.hypot(tangent, 1.0)
                sine = tangent * cosine
                for k in range(size):
                    above, below = a[i][k], a[j][k]
                    a[i][k] = cosine * above - sine * below
                    a[j][k] = sine * above + cosine * below
                for k in range(size):
                    left, right = a[k][i], a[k][j]
                    a[k][i] = cosine * left - sine * right
                    a[k][j] = sine * left + cosine * right
                for k in range(size):
                    left, right = vectors[k][i], vectors[k][j]
                    vectors[k][i] = cosine * left - sine * right
                    vectors[k][j] = sine * left + cosine * right
    pairs = sorted(((a[i][i], [vectors[row][i] for row in range(size)]) for i in range(size)), key=lambda item: -item[0])
    values = [value for value, _ in pairs]
    components = [component for _, component in pairs]
    for component in components:
        peak = max(range(size), key=lambda i: abs(component[i]))
        if component[peak] < 0:
            for i in range(size):
                component[i] = -component[i]
    return values, components


def _project(vector, components):
    scores = [sum(component[i] * vector[i] for i in range(len(vector))) for component in components]
    residual = list(vector)
    for score, component in zip(scores, components):
        for i in range(len(residual)):
            residual[i] -= score * component[i]
    spe = sum(value * value for value in residual)
    return scores, residual, spe


def _t2(scores, eigenvalues):
    return sum(score * score / eigenvalue for score, eigenvalue in zip(scores, eigenvalues))


def _quantile(values, quantile):
    ordered = sorted(values)
    size = len(ordered)
    rank = min(size, max(1, math.ceil(quantile * size)))
    threshold = ordered[rank - 1]
    exceedances = sum(1 for value in values if value > threshold)
    return {'quantile_requested': quantile, 'samples': size, 'rank': rank, 'effective_quantile': rank / size, 'value': threshold, 'calibration_exceedances': exceedances, 'calibration_exceedance_rate': exceedances / size, 'finest_resolvable_tail': 1 / size, 'tail_precision_limited': (1 - quantile) * size < 1 - 1e-9}


def _declaration_state(store, calendar, contract_id, period):
    if store.one("SELECT name FROM sqlite_master WHERE type='table' AND name='declarations'") is None:
        return 'undeclared_period'
    declarations = {row['side']: row for row in store.all('SELECT * FROM declarations WHERE contract_id=? AND period=?', (contract_id, period))}
    for side in ('provider', 'consumer'):
        declaration = declarations.get(side)
        if declaration is None:
            return 'undeclared_period'
        current = calendar.side_manifest(contract_id, period, side)
        if current['pending_events'] or current['sequence'] != declaration['sequence'] or current['usage_count'] != declaration['usage_count'] or current['digest'] != declaration['digest']:
            return 'declaration_stale'
    return 'declared'


def _window(store, calendar, contract, tzinfo, first, last, knowledge, meters):
    contract_id = contract['contract_id']
    quantities = {}
    lineage = {}
    disputed = set()
    for period in _months(first, last):
        for identity in reconcile.identities(store, contract_id, period):
            row = reconcile.classify(store, contract_id, identity, period)
            if row['status'] == 'disputed':
                for side in ('provider', 'consumer'):
                    evidence = row['evidence'][side]
                    if evidence is not None:
                        disputed.add(_local_day(evidence['payload']['at'], tzinfo))
            chosen = row['chosen']
            if chosen is not None:
                day = _local_day(chosen['at'], tzinfo)
                meter = chosen['meter']
                cell = quantities.setdefault(day, {item: Decimal(0) for item in meters})
                cell[meter] += Decimal(chosen['quantity'])
                lineage.setdefault(day, {}).setdefault(meter, []).append({'usage_id': identity, 'base_hash': row['base_hash'], 'quantity': chosen['quantity']})
    declarations = {period: _declaration_state(store, calendar, contract_id, period) for period in _months(first, last)}
    valid_from = _parse_instant(contract['definition']['valid_from'])
    valid_until = _parse_instant(contract['definition']['valid_until'])
    rows = []
    excluded = []
    day = first
    while day <= last:
        stamp = day.isoformat()
        day_start, day_end = _day_bounds(day, tzinfo)
        reason = None
        if day_end <= valid_from or day_start >= valid_until:
            reason = 'outside_contract'
        elif stamp in disputed:
            reason = 'disputed'
        elif day_end > knowledge:
            reason = 'beyond_cutoff'
        elif declarations[stamp[:7]] != 'declared':
            reason = declarations[stamp[:7]]
        if reason:
            excluded.append({'day': stamp, 'reason': reason})
        else:
            cell = quantities.get(stamp, {item: Decimal(0) for item in meters})
            rows.append({'day': stamp, 'quantities': {meter: cell[meter] for meter in meters}, 'lineage': lineage.get(stamp, {})})
        day += datetime.timedelta(days=1)
    return rows, excluded


class Monitor:
    def __init__(self, store):
        self.store = store
        if store.depth:
            reject('monitor_transaction', 'Initialize usage monitor outside transactions')
        store.db.executescript(DDL)
        self.calendar = object.__new__(Calendar)
        self.calendar.store = store

    def fit(self, model_id, contract_id, timezone, train_start, train_end, calibration_start, calibration_end, knowledge_cutoff, quantile=0.95, variance_floor=0.8):
        key(model_id)
        contract = registry.get(self.store, contract_id)
        tzinfo = _timezone(timezone)
        first_train = _date(train_start, 'train_start')
        last_train = _date(train_end, 'train_end')
        first_calibration = _date(calibration_start, 'calibration_start')
        last_calibration = _date(calibration_end, 'calibration_end')
        if not first_train <= last_train < first_calibration <= last_calibration:
            reject('monitor_periods', 'Train and calibration windows must be ordered and disjoint in time')
        for first, last in ((first_train, last_train), (first_calibration, last_calibration)):
            if (last - first).days + 1 > MAX_RANGE_DAYS:
                reject('monitor_range', 'Window exceeds the bounded day range', maximum=MAX_RANGE_DAYS)
        cutoff = _parse_instant(knowledge_cutoff)
        level = _fraction(quantile, 'monitor_quantile', 0.8, 0.999)
        floor = _fraction(variance_floor, 'monitor_floor', 0.5, 0.999)
        meters = list(contract['definition']['meters'])
        if len(meters) < 2:
            reject('monitor_meters', 'Multivariate monitoring needs at least two meters; anomaly.outliers covers a single meter')
        train_rows, train_excluded = _window(self.store, self.calendar, contract, tzinfo, first_train, last_train, cutoff, meters)
        calibration_rows, calibration_excluded = _window(self.store, self.calendar, contract, tzinfo, first_calibration, last_calibration, cutoff, meters)
        if len(train_rows) < MIN_TRAIN_DAYS:
            reject('monitor_train_days', 'Too few confirmed-complete training days', eligible=len(train_rows), required=MIN_TRAIN_DAYS, excluded=train_excluded)
        if len(calibration_rows) < MIN_CALIBRATION_DAYS:
            reject('monitor_calibration_days', 'Too few confirmed-complete calibration days', eligible=len(calibration_rows), required=MIN_CALIBRATION_DAYS, excluded=calibration_excluded)
        size = len(train_rows)
        width = len(meters)
        validation_size = max(2, size // 4)
        fit_size = size - validation_size
        if fit_size < max(3, width):
            reject('monitor_train_days', 'Training split leaves too few fit days for the meter count', fit_days=fit_size, meters=width)
        matrix = [[float(row['quantities'][meter]) for meter in meters] for row in train_rows]
        fit_matrix, validation_matrix = matrix[:fit_size], matrix[fit_size:]
        fit_scaling = _fit_scaling(fit_matrix)
        fit_values, fit_vectors = _eigh(_covariance([_standardize(row, fit_scaling) for row in fit_matrix]))
        total_variance = sum(fit_values)
        rank = sum(1 for value in fit_values if value > fit_values[0] * EIGENVALUE_TOLERANCE) if fit_values[0] > 0 else 0
        if rank == 0:
            reject('monitor_degenerate', 'Training matrix carries no variance')
        validation_z = [_standardize(row, fit_scaling) for row in validation_matrix]
        candidates = []
        for k in range(1, min(rank, width - 1) + 1):
            explained = sum(fit_values[:k]) / total_variance
            forward = sum(_project(z, fit_vectors[:k])[2] for z in validation_z) / len(validation_z) / width
            candidates.append({'k': k, 'train_explained_variance': explained, 'forward_spe_mean': forward})
        eligible = [candidate for candidate in candidates if candidate['train_explained_variance'] >= floor]
        if not eligible:
            reject('monitor_no_candidate', 'No candidate component count reaches the explained-variance floor', candidates=candidates, variance_floor=floor)
        chosen = min(eligible, key=lambda candidate: (candidate['forward_spe_mean'], candidate['k']))
        for candidate in candidates:
            candidate['selected'] = candidate['k'] == chosen['k']
        k = chosen['k']
        scaling = _fit_scaling(matrix)
        final_values, final_vectors = _eigh(_covariance([_standardize(row, scaling) for row in matrix]))
        final_rank = sum(1 for value in final_values if value > final_values[0] * EIGENVALUE_TOLERANCE) if final_values[0] > 0 else 0
        if k > final_rank:
            reject('monitor_degenerate', 'Full-window training covariance lost rank against the fit split')
        components = final_vectors[:k]
        component_values = final_values[:k]
        calibration_statistics = []
        for row in calibration_rows:
            vector = [float(row['quantities'][meter]) for meter in meters]
            scores, _, spe = _project(_standardize(vector, scaling), components)
            calibration_statistics.append({'day': row['day'], 't2': _t2(scores, component_values), 'spe': spe})
        t2_threshold = _quantile([row['t2'] for row in calibration_statistics], level)
        spe_threshold = _quantile([row['spe'] for row in calibration_statistics], level)
        joint = sum(1 for row in calibration_statistics if row['t2'] > t2_threshold['value'] and row['spe'] > spe_threshold['value'])
        covariance = _covariance([_standardize(row, scaling) for row in matrix])
        diagnostics = {
            'constant_columns': [meter for meter, entry in zip(meters, scaling) if entry['constant']],
            'condition_number': final_values[0] / final_values[-1] if final_values[-1] > 0 else None,
            'near_collinear': final_values[-1] <= final_values[0] * EIGENVALUE_TOLERANCE,
            'high_correlation_pairs': [{'meters': [meters[i], meters[j]], 'correlation': covariance[i][j]} for i in range(width) for j in range(i + 1, width) if abs(covariance[i][j]) >= 0.999],
            'scale_spread': (lambda scales: max(scales) / min(scales) if scales else None)([entry['std'] for entry in scaling if not entry['constant']]),
            'scale_warning': False,
            'samples_per_meter': fit_size / width,
            'high_dimensional_small_sample': fit_size < 5 * width,
            'singular_training_covariance': final_rank < width,
            'zero_filled_days': {'train': sum(1 for row in train_rows if not row['lineage']), 'calibration': sum(1 for row in calibration_rows if not row['lineage'])},
            'excluded_days': {'train': train_excluded, 'calibration': calibration_excluded},
        }
        diagnostics['scale_warning'] = diagnostics['scale_spread'] is not None and diagnostics['scale_spread'] > 1e3
        lineage_rows = []
        for row in train_rows + calibration_rows:
            for meter in meters:
                evidence = sorted(row['lineage'].get(meter, []), key=lambda entry: entry['usage_id'])
                if evidence:
                    lineage_rows.append({'day': row['day'], 'meter': meter, 'quantity': encoded(row['quantities'][meter]), 'evidence': evidence})
        payload = {
            'format': 'meterbook-usage-monitor',
            'version': 1,
            'model_id': model_id,
            'contract_id': contract_id,
            'frozen': {
                'contract_revision': contract['revision'],
                'meters': meters,
                'units': 'contract_defined_quantity_per_meter',
                'timezone': timezone,
                'day_boundary': 'local_midnight',
                'knowledge_cutoff': instant(knowledge_cutoff),
                'periods': {'train': {'start': train_start, 'end': train_end}, 'calibration': {'start': calibration_start, 'end': calibration_end}},
            },
            'scaling': {meter: scaling[index] for index, meter in enumerate(meters)},
            'pca': {
                'chosen_k': k,
                'rank': final_rank,
                'eigenvalues': final_values,
                'components': [{'index': index + 1, 'eigenvalue': component_values[index], 'loadings': {meters[column]: components[index][column] for column in range(width)}} for index in range(k)],
                'train_explained_variance': sum(component_values) / sum(final_values),
                'candidates': candidates,
                'selection': {'variance_floor': floor, 'rule': 'among candidates meeting the training explained-variance floor choose the lowest forward reconstruction error on the held-back validation tail, ties prefer fewer components', 'fit_days': fit_size, 'validation_days': validation_size},
            },
            'thresholds': {'quantile': level, 't2': t2_threshold, 'spe': spe_threshold, 'joint': {'calibration_exceedances': joint, 'calibration_exceedance_rate': joint / len(calibration_statistics)}, 'note': 'empirical quantiles frozen from the independent calibration window; not a guaranteed long-run error rate'},
            'train_matrix': {'days': [row['day'] for row in train_rows], 'values': [[encoded(row['quantities'][meter]) for meter in meters] for row in train_rows]},
            'calibration_statistics': calibration_statistics,
            'diagnostics': diagnostics,
            'lineage_digest': digest(lineage_rows),
            'effect': 'diagnostic_only_no_billing_mutation',
        }
        with self.store.atomic():
            if self.store.one('SELECT model_id FROM monitor_models WHERE model_id=?', (model_id,)):
                reject('monitor_exists', 'Monitor model identifiers are immutable')
            version = (self.store.scalar('SELECT MAX(version) FROM monitor_models WHERE contract_id=?', (contract_id,)) or 0) + 1
            payload['model_version'] = version
            self.store.execute('INSERT INTO monitor_models VALUES(?,?,?,?,?,?)', (model_id, contract_id, version, canonical(payload), digest(payload), self.store.clock()))
            for row in lineage_rows:
                self.store.execute('INSERT INTO monitor_lineage VALUES(?,?,?,?,?)', (model_id, row['day'], row['meter'], row['quantity'], canonical(row['evidence'])))
            self.store.log('monitor.fitted', {'model_id': model_id, 'contract_id': contract_id, 'version': version, 'chosen_k': k, 'train_days': size, 'calibration_days': len(calibration_rows)})
        return self.get(model_id)

    def get(self, model_id):
        key(model_id)
        row = self.store.one('SELECT * FROM monitor_models WHERE model_id=?', (model_id,))
        if row is None:
            reject('monitor_missing', 'Monitor model does not exist')
        payload = json.loads(row['payload'])
        if digest(payload) != row['digest']:
            reject('monitor_corrupt', 'Stored monitor model differs from its checksum')
        lineage_rows = [{'day': item['day'], 'meter': item['meter'], 'quantity': item['quantity'], 'evidence': json.loads(item['evidence'])} for item in self.store.all('SELECT * FROM monitor_lineage WHERE model_id=? ORDER BY day,meter', (model_id,))]
        if digest(lineage_rows) != payload['lineage_digest']:
            reject('monitor_corrupt', 'Stored monitor lineage differs from its checksum')
        lineage = defaultdict(dict)
        for item in lineage_rows:
            lineage[item['day']][item['meter']] = {'quantity': item['quantity'], 'evidence': item['evidence']}
        return {'model_id': model_id, 'contract_id': row['contract_id'], 'version': row['version'], 'created_at': row['created_at'], 'payload': payload, 'lineage': dict(lineage), 'verified': True}

    def versions(self, contract_id):
        registry.get(self.store, contract_id)
        result = []
        for row in self.store.all('SELECT * FROM monitor_models WHERE contract_id=? ORDER BY version', (contract_id,)):
            payload = json.loads(row['payload'])
            if digest(payload) != row['digest']:
                reject('monitor_corrupt', 'Stored monitor model differs from its checksum')
            result.append({'model_id': row['model_id'], 'version': row['version'], 'created_at': row['created_at'], 'chosen_k': payload['pca']['chosen_k'], 'meters': payload['frozen']['meters'], 'periods': payload['frozen']['periods']})
        return result

    def scan(self, model_id, scan_start, scan_end):
        model = self.get(model_id)
        payload = model['payload']
        frozen = payload['frozen']
        contract = registry.get(self.store, payload['contract_id'])
        meters = frozen['meters']
        if contract['definition']['meters'] != meters:
            reject('monitor_columns_changed', 'Contract meter set differs from the frozen model columns; fit a new model version instead of reusing this one', expected=meters, actual=contract['definition']['meters'])
        first = _date(scan_start, 'scan_start')
        last = _date(scan_end, 'scan_end')
        if not first <= last or (last - first).days + 1 > MAX_RANGE_DAYS:
            reject('monitor_range', 'Scan window must be a bounded ordered day range', maximum=MAX_RANGE_DAYS)
        if first <= _date(frozen['periods']['calibration']['end'], 'calibration_end'):
            reject('monitor_period_overlap', 'Monitoring window must start after the frozen calibration window', calibration_end=frozen['periods']['calibration']['end'])
        tzinfo = _timezone(frozen['timezone'])
        knowledge = _parse_instant(self.store.clock())
        rows, excluded = _window(self.store, self.calendar, contract, tzinfo, first, last, knowledge, meters)
        scaling = [payload['scaling'][meter] for meter in meters]
        components = [[component['loadings'][meter] for meter in meters] for component in payload['pca']['components']]
        eigenvalues = [component['eigenvalue'] for component in payload['pca']['components']]
        t2_limit = payload['thresholds']['t2']['value']
        spe_limit = payload['thresholds']['spe']['value']
        days = []
        for row in rows:
            vector = [float(row['quantities'][meter]) for meter in meters]
            z = _standardize(vector, scaling)
            scores, residual, spe = _project(z, components)
            t2 = _t2(scores, eigenvalues)
            t2_exceeded = t2 > t2_limit
            spe_exceeded = spe > spe_limit
            alert = 'joint' if t2_exceeded and spe_exceeded else 't2' if t2_exceeded else 'spe' if spe_exceeded else 'none'
            detail = {}
            for index, meter in enumerate(meters):
                detail[meter] = {
                    'observed': encoded(row['quantities'][meter]),
                    'expected': scaling[index]['mean'] + scaling[index]['scale'] * (z[index] - residual[index]),
                    'standardized_deviation': z[index],
                    'residual': residual[index],
                    'spe_contribution': residual[index] ** 2,
                    'spe_share': (residual[index] ** 2 / spe) if spe > 0 else 0.0,
                    'usage_ids': [entry['usage_id'] for entry in row['lineage'].get(meter, [])],
                }
            days.append({'day': row['day'], 't2': t2, 'spe': spe, 'scores': scores, 't2_exceeded': t2_exceeded, 'spe_exceeded': spe_exceeded, 'alert': alert, 'interpretation': INTERPRETATIONS[alert], 'meters': detail})
        alerts = [row for row in days if row['alert'] != 'none']
        return {
            'model_id': model_id,
            'model_version': payload['model_version'],
            'contract_id': payload['contract_id'],
            'window': {'start': scan_start, 'end': scan_end, 'knowledge': instant(self.store.clock())},
            'thresholds': {'t2': t2_limit, 'spe': spe_limit, 'quantile': payload['thresholds']['quantile']},
            'calibration_performance': {
                'samples': payload['thresholds']['t2']['samples'],
                't2_exceedance_rate': payload['thresholds']['t2']['calibration_exceedance_rate'],
                'spe_exceedance_rate': payload['thresholds']['spe']['calibration_exceedance_rate'],
                'joint_exceedance_rate': payload['thresholds']['joint']['calibration_exceedance_rate'],
                'note': 'empirical rates observed on the frozen calibration window at fit time; they describe that window only and are not a guaranteed long-run error rate',
            },
            'alert_semantics': ALERT_SEMANTICS,
            'days': days,
            'excluded': excluded,
            'alerts': [{'day': row['day'], 'alert': row['alert'], 't2': row['t2'], 'spe': row['spe']} for row in alerts],
            'summary': {'days_evaluated': len(days), 'days_excluded': len(excluded), 'alerts': len(alerts), 'joint_alerts': sum(1 for row in alerts if row['alert'] == 'joint')},
            'contract_revision_warning': contract['revision'] != frozen['contract_revision'],
            'effect': 'diagnostic_only_no_billing_mutation',
        }
