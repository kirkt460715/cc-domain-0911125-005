import math
import unittest
from tests.support import Case, definition, event, usage
from meterbook.audit import monitor, anomalies
from meterbook.contracts import registry
from meterbook.events import receiver
from meterbook.values import MeterError


class Algebra(unittest.TestCase):
    def test_eigh_recovers_known_spectrum(self):
        values, vectors = monitor._eigh([[2.0, 1.0], [1.0, 2.0]])
        root = 1 / math.sqrt(2)
        self.assertAlmostEqual(values[0], 3.0)
        self.assertAlmostEqual(values[1], 1.0)
        self.assertAlmostEqual(vectors[0][0], root)
        self.assertAlmostEqual(vectors[0][1], root)
        self.assertAlmostEqual(vectors[1][0], root)
        self.assertAlmostEqual(vectors[1][1], -root)
        diagonal, axes = monitor._eigh([[3.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]])
        self.assertEqual(diagonal, [3.0, 2.0, 1.0])
        self.assertEqual(axes[0], [1.0, 0.0, 0.0])
        self.assertEqual(axes[2], [0.0, 0.0, 1.0])

    def test_projection_and_residual_sum_of_squares(self):
        rows = [[1.0, 2.0], [2.0, 4.0], [3.0, 6.0], [4.0, 8.0]]
        scaling = monitor._fit_scaling(rows)
        standardized = [monitor._standardize(row, scaling) for row in rows]
        values, vectors = monitor._eigh(monitor._covariance(standardized))
        self.assertAlmostEqual(values[0], 2.0)
        self.assertAlmostEqual(values[1], 0.0)
        probe = [3.0, -1.0]
        scores, residual, spe = monitor._project(probe, vectors[:1])
        self.assertAlmostEqual(scores[0], math.sqrt(2))
        self.assertAlmostEqual(residual[0], 2.0)
        self.assertAlmostEqual(residual[1], -2.0)
        self.assertAlmostEqual(spe, 8.0)
        direct = sum((probe[i] - scores[0] * vectors[0][i]) ** 2 for i in range(2))
        self.assertAlmostEqual(spe, direct)
        pythagoras = sum(score * score for score in scores) + spe
        self.assertAlmostEqual(sum(value * value for value in probe), pythagoras)

    def test_quantile_reports_achievable_tail_precision(self):
        values = [float(index) for index in range(1, 11)]
        fine = monitor._quantile(values, 0.9)
        self.assertEqual(fine['value'], 9.0)
        self.assertEqual(fine['rank'], 9)
        self.assertEqual(fine['effective_quantile'], 0.9)
        self.assertEqual(fine['calibration_exceedances'], 1)
        self.assertFalse(fine['tail_precision_limited'])
        coarse = monitor._quantile(values, 0.95)
        self.assertEqual(coarse['value'], 10.0)
        self.assertEqual(coarse['effective_quantile'], 1.0)
        self.assertEqual(coarse['calibration_exceedances'], 0)
        self.assertEqual(coarse['finest_resolvable_tail'], 0.1)
        self.assertTrue(coarse['tail_precision_limited'])

    def test_timezone_parsing_and_local_day(self):
        self.assertEqual(monitor._local_day('2026-08-05T23:30:00.000000Z', monitor._timezone('UTC')), '2026-08-05')
        self.assertEqual(monitor._local_day('2026-08-05T23:30:00.000000Z', monitor._timezone('UTC+08:00')), '2026-08-06')
        self.assertEqual(monitor._local_day('2026-08-05T16:30:00.000000Z', monitor._timezone('-05:00')), '2026-08-05')
        with self.assertRaises(MeterError) as caught:
            monitor._timezone('GMT')
        self.assertEqual(caught.exception.code, 'monitor_timezone')


class MonitorCase(Case):
    def setUp(self):
        super().setUp()
        registry.create(self.store, 'contract-m', 'provider-m', 'consumer-m', 'CNY', definition(meters=['downloads', 'queries'], tolerance={'downloads': '0.5', 'queries': '0.5'}))
        self.streams = {'provider': 0, 'consumer': 0}

    def report(self, day, downloads, queries, contract='contract-m'):
        for meter, quantity in (('downloads', downloads), ('queries', queries)):
            identity = f'u-{day}-{meter}'
            for side in ('provider', 'consumer'):
                self.streams[side] += 1
                receiver.receive(self.store, event(side, self.streams[side], identity, 1, 'report', contract_id=contract, usage=usage(str(quantity), meter=meter, at=f'{day}T12:00:00Z')))
                self.assertIsNone(receiver.drain(self.store, contract, side)['blocked'])

    def report_mismatch(self, day, provider_quantity, consumer_quantity):
        for side, quantity in (('provider', provider_quantity), ('consumer', consumer_quantity)):
            self.streams[side] += 1
            receiver.receive(self.store, event(side, self.streams[side], f'u-{day}-clash', 1, 'report', contract_id='contract-m', usage=usage(str(quantity), meter='downloads', at=f'{day}T12:00:00Z')))
            self.assertIsNone(receiver.drain(self.store, 'contract-m', side)['blocked'])

    def declare(self, *periods, contract='contract-m'):
        for period in periods:
            self.app.calendar.open(contract, period)
            for side in ('provider', 'consumer'):
                manifest = self.app.calendar.side_manifest(contract, period, side)
                self.app.calendar.declare(contract, period, side, 0, manifest['sequence'], manifest['usage_count'], manifest['digest'])

    def build_data(self):
        for day in list(range(1, 10)) + list(range(11, 32)):
            downloads = 100 + 3 * day + (day % 3)
            self.report(f'2026-08-{day:02d}', downloads, 2 * downloads + (1 if day % 2 else -1))
        self.declare('2026-08')

    def fit_model(self, model_id='model-a', **overrides):
        settings = dict(model_id=model_id, contract_id='contract-m', timezone='UTC', train_start='2026-08-01', train_end='2026-08-20', calibration_start='2026-08-21', calibration_end='2026-08-31', knowledge_cutoff='2026-09-01T00:00:00Z')
        settings.update(overrides)
        return self.app.monitor.fit(**settings)


class Monitoring(MonitorCase):
    def test_joint_monitoring_distinguishes_magnitude_and_relation_shifts(self):
        self.build_data()
        model = self.fit_model()
        payload = model['payload']
        self.assertEqual(payload['pca']['chosen_k'], 1)
        self.assertEqual(payload['frozen']['meters'], ['downloads', 'queries'])
        self.report('2026-09-01', 170, 340)
        self.report('2026-09-02', 500, 1000)
        self.report('2026-09-03', 170, 100)
        self.report('2026-09-04', 500, 100)
        self.declare('2026-09')
        result = self.app.monitor.scan('model-a', '2026-09-01', '2026-09-04')
        days = {row['day']: row for row in result['days']}
        normal = days['2026-09-01']
        self.assertEqual(normal['alert'], 'none')
        self.assertEqual(normal['interpretation'], 'in_control')
        scaled = days['2026-09-02']
        self.assertEqual(scaled['alert'], 't2')
        self.assertTrue(scaled['t2_exceeded'])
        self.assertFalse(scaled['spe_exceeded'])
        self.assertEqual(scaled['interpretation'], 'magnitude_shift_inside_learned_relation')
        broken = days['2026-09-03']
        self.assertEqual(broken['alert'], 'spe')
        self.assertTrue(broken['spe_exceeded'])
        self.assertFalse(broken['t2_exceeded'])
        self.assertEqual(broken['interpretation'], 'learned_meter_relation_broken')
        joint = days['2026-09-04']
        self.assertEqual(joint['alert'], 'joint')
        self.assertTrue(joint['t2_exceeded'])
        self.assertTrue(joint['spe_exceeded'])
        queries = broken['meters']['queries']
        self.assertLess(queries['residual'], 0)
        self.assertGreater(broken['meters']['downloads']['residual'], 0)
        self.assertGreater(queries['spe_share'], 0.4)
        self.assertEqual(queries['observed'], '100')
        self.assertGreater(queries['standardized_deviation'], -3)
        self.assertIn('expected', queries)
        self.assertEqual(queries['usage_ids'], ['u-2026-09-03-queries'])
        self.assertEqual(result['summary']['joint_alerts'], 1)
        self.assertIn('not a guaranteed long-run error rate', result['calibration_performance']['note'])
        self.assertIn('never billing judgements', result['alert_semantics']['note'])
        self.assertEqual(result['effect'], 'diagnostic_only_no_billing_mutation')

    def test_confirmed_empty_day_is_zero_filled_with_lineage(self):
        self.build_data()
        model = self.fit_model()
        payload = model['payload']
        index = payload['train_matrix']['days'].index('2026-08-10')
        self.assertEqual(payload['train_matrix']['values'][index], ['0', '0'])
        self.assertEqual(payload['diagnostics']['zero_filled_days']['train'], 1)
        self.assertNotIn('2026-08-10', model['lineage'])
        first = model['lineage']['2026-08-01']
        self.assertEqual(first['downloads']['quantity'], '104')
        self.assertEqual(first['downloads']['evidence'][0]['usage_id'], 'u-2026-08-01-downloads')
        self.assertEqual(len(first['downloads']['evidence'][0]['base_hash']), 64)

    def test_disputed_and_undeclared_days_are_never_zero_filled(self):
        self.build_data()
        self.fit_model()
        self.report('2026-09-01', 170, 340)
        self.report_mismatch('2026-09-02', 100, 999)
        self.report('2026-09-03', 175, 350)
        self.declare('2026-09')
        result = self.app.monitor.scan('model-a', '2026-09-01', '2026-10-02')
        evaluated = {row['day'] for row in result['days']}
        self.assertIn('2026-09-01', evaluated)
        self.assertIn('2026-09-03', evaluated)
        self.assertNotIn('2026-09-02', evaluated)
        reasons = {row['day']: row['reason'] for row in result['excluded']}
        self.assertEqual(reasons['2026-09-02'], 'disputed')
        self.assertEqual(reasons['2026-10-01'], 'undeclared_period')
        self.assertEqual(reasons['2026-10-02'], 'undeclared_period')

    def test_stale_declaration_blocks_confirmation(self):
        self.build_data()
        self.fit_model()
        self.report('2026-09-01', 170, 340)
        self.declare('2026-09')
        self.report('2026-09-02', 175, 350)
        result = self.app.monitor.scan('model-a', '2026-09-01', '2026-09-02')
        self.assertEqual(result['days'], [])
        self.assertEqual({row['reason'] for row in result['excluded']}, {'declaration_stale'})

    def test_pending_event_marks_declaration_stale(self):
        self.build_data()
        self.fit_model()
        self.report('2026-09-01', 170, 340)
        self.declare('2026-09')
        self.streams['provider'] += 1
        receiver.receive(self.store, event('provider', self.streams['provider'], 'u-2026-09-01-downloads', 2, 'correct', contract_id='contract-m', usage=usage('171', meter='downloads', at='2026-09-01T12:00:00Z')))
        result = self.app.monitor.scan('model-a', '2026-09-01', '2026-09-01')
        self.assertEqual(result['days'], [])
        self.assertEqual(result['excluded'][0]['reason'], 'declaration_stale')

    def test_thresholds_frozen_from_calibration_and_scan_never_retrains(self):
        self.build_data()
        model = self.fit_model()
        payload = model['payload']
        threshold = payload['thresholds']['t2']
        self.assertEqual(threshold['samples'], 11)
        self.assertEqual(threshold['effective_quantile'], 1.0)
        self.assertAlmostEqual(threshold['finest_resolvable_tail'], 1 / 11)
        self.assertTrue(threshold['tail_precision_limited'])
        self.assertEqual(threshold['calibration_exceedances'], 0)
        self.report('2026-09-01', 500, 1000)
        self.declare('2026-09')
        before = self.store.scalar('SELECT COUNT(*) FROM monitor_models')
        digest_before = self.store.scalar("SELECT digest FROM monitor_models WHERE model_id='model-a'")
        result = self.app.monitor.scan('model-a', '2026-09-01', '2026-09-01')
        self.assertEqual(result['days'][0]['alert'], 't2')
        self.assertEqual(result['thresholds']['t2'], threshold['value'])
        self.assertEqual(result['thresholds']['spe'], payload['thresholds']['spe']['value'])
        self.assertEqual(self.store.scalar('SELECT COUNT(*) FROM monitor_models'), before)
        self.assertEqual(self.store.scalar("SELECT digest FROM monitor_models WHERE model_id='model-a'"), digest_before)
        self.assertEqual(self.store.scalar('SELECT COUNT(*) FROM audit WHERE action=?', ('monitor.fitted',)), 1)

    def test_column_set_change_requires_new_model_version(self):
        self.build_data()
        self.fit_model()
        registry.revise(self.store, 'contract-m', 1, definition(meters=['downloads', 'queries', 'storage'], tolerance={'downloads': '0.5', 'queries': '0.5'}), 'add storage meter')
        self.report('2026-09-01', 170, 340)
        self.declare('2026-09')
        self.error('monitor_columns_changed', self.app.monitor.scan, 'model-a', '2026-09-01', '2026-09-01')
        self.assertEqual(len(self.app.monitor.versions('contract-m')), 1)

    def test_explicit_retrain_creates_version_and_models_stay_immutable(self):
        self.build_data()
        self.fit_model()
        self.fit_model('model-b')
        versions = self.app.monitor.versions('contract-m')
        self.assertEqual([row['version'] for row in versions], [1, 2])
        self.assertEqual([row['model_id'] for row in versions], ['model-a', 'model-b'])
        self.error('monitor_exists', self.fit_model)
        self.report('2026-09-01', 170, 340)
        self.declare('2026-09')
        result = self.app.monitor.scan('model-a', '2026-09-01', '2026-09-01')
        self.assertEqual(result['model_version'], 1)
        self.assertEqual(result['days'][0]['alert'], 'none')
        self.assertEqual(self.store.scalar('SELECT COUNT(*) FROM monitor_models'), 2)

    def test_model_and_lineage_tamper_detected(self):
        self.build_data()
        self.fit_model()
        self.fit_model('model-b')
        self.assertTrue(self.app.monitor.get('model-a')['verified'])
        row = self.store.one("SELECT payload FROM monitor_models WHERE model_id='model-a'")
        self.store.execute("UPDATE monitor_models SET payload=? WHERE model_id='model-a'", (row['payload'].replace('"chosen_k":1', '"chosen_k":2'),))
        self.error('monitor_corrupt', self.app.monitor.get, 'model-a')
        self.store.execute("UPDATE monitor_lineage SET quantity='0' WHERE model_id='model-b' AND day='2026-08-01'")
        self.error('monitor_corrupt', self.app.monitor.get, 'model-b')
        self.error('monitor_missing', self.app.monitor.get, 'model-z')

    def test_knowledge_cutoff_excludes_unclosed_days(self):
        self.build_data()
        model = self.fit_model(knowledge_cutoff='2026-08-30T00:00:00Z')
        excluded = model['payload']['diagnostics']['excluded_days']['calibration']
        self.assertEqual({row['day'] for row in excluded}, {'2026-08-30', '2026-08-31'})
        self.assertEqual({row['reason'] for row in excluded}, {'beyond_cutoff'})
        self.error('monitor_calibration_days', self.fit_model, 'model-c', knowledge_cutoff='2026-08-25T00:00:00Z')

    def test_undeclared_periods_block_training(self):
        for day in range(1, 10):
            self.report(f'2026-08-{day:02d}', 100 + day, 200 + 2 * day)
        caught = self.error('monitor_train_days', self.fit_model, 'model-a', train_end='2026-08-08', calibration_start='2026-08-09', calibration_end='2026-08-09')
        self.assertEqual(caught.details['eligible'], 0)

    def test_diagnostics_cover_constant_collinear_and_small_sample(self):
        registry.create(self.store, 'contract-c', 'provider-c', 'consumer-c', 'CNY', definition(meters=['downloads', 'queries', 'storage'], tolerance={'downloads': '0.5', 'queries': '0.5', 'storage': '0.5'}))
        streams = {'provider': 0, 'consumer': 0}
        for day in range(1, 14):
            for meter, quantity in (('downloads', 100 + 3 * day), ('queries', 200 + 6 * day), ('storage', 7)):
                for side in ('provider', 'consumer'):
                    streams[side] += 1
                    receiver.receive(self.store, event(side, streams[side], f'c-{day}-{meter}', 1, 'report', contract_id='contract-c', usage=usage(str(quantity), meter=meter, at=f'2026-08-{day:02d}T12:00:00Z')))
                    self.assertIsNone(receiver.drain(self.store, 'contract-c', side)['blocked'])
        self.declare('2026-08', contract='contract-c')
        model = self.app.monitor.fit('model-c', 'contract-c', 'UTC', '2026-08-01', '2026-08-08', '2026-08-09', '2026-08-13', '2026-08-14T00:00:00Z')
        diagnostics = model['payload']['diagnostics']
        self.assertEqual(diagnostics['constant_columns'], ['storage'])
        self.assertTrue(diagnostics['near_collinear'])
        self.assertIsNone(diagnostics['condition_number'])
        self.assertEqual(diagnostics['high_correlation_pairs'][0]['meters'], ['downloads', 'queries'])
        self.assertTrue(diagnostics['singular_training_covariance'])
        self.assertTrue(diagnostics['high_dimensional_small_sample'])
        self.assertEqual(diagnostics['samples_per_meter'], 2.0)
        scaling = model['payload']['scaling']
        self.assertTrue(scaling['storage']['constant'])
        self.assertEqual(scaling['storage']['scale'], 1.0)
        self.assertEqual(model['payload']['pca']['chosen_k'], 1)

    def test_fit_validation_and_window_separation(self):
        self.error('monitor_meters', self.app.monitor.fit, 'model-1', 'contract-a', 'UTC', '2026-08-01', '2026-08-10', '2026-08-11', '2026-08-20', '2026-09-01T00:00:00Z')
        self.error('monitor_periods', self.app.monitor.fit, 'model-1', 'contract-m', 'UTC', '2026-08-01', '2026-08-15', '2026-08-10', '2026-08-20', '2026-09-01T00:00:00Z')
        self.error('monitor_quantile', self.app.monitor.fit, 'model-1', 'contract-m', 'UTC', '2026-08-01', '2026-08-10', '2026-08-11', '2026-08-20', '2026-09-01T00:00:00Z', 0.5)
        self.error('monitor_floor', self.app.monitor.fit, 'model-1', 'contract-m', 'UTC', '2026-08-01', '2026-08-10', '2026-08-11', '2026-08-20', '2026-09-01T00:00:00Z', 0.95, 2)
        self.error('monitor_day', self.app.monitor.fit, 'model-1', 'contract-m', 'UTC', '2026-08-01', '2026-08-40', '2026-08-11', '2026-08-20', '2026-09-01T00:00:00Z')
        self.build_data()
        self.fit_model()
        self.error('monitor_period_overlap', self.app.monitor.scan, 'model-a', '2026-08-21', '2026-09-03')
        self.error('monitor_range', self.app.monitor.scan, 'model-a', '2026-09-03', '2026-09-01')

    def test_univariate_outliers_entry_still_available(self):
        for index, quantity in enumerate(['10', '10', '10', '100'], 1):
            self.pair(quantity, index, identity=f'usage-{index}')
        report = anomalies.outliers(self.store, 'contract-a', '2026-11')
        self.assertEqual(report['reports'][0]['flagged'], [{'usage_id': 'usage-4', 'quantity': '100'}])
        operations = [row['operation'] for row in self.app.capabilities()]
        self.assertIn('anomaly.outliers', operations)
        self.assertIn('monitor.fit', operations)
        self.assertIn('monitor.scan', operations)
