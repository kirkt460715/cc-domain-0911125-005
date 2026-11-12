# 数据用量核对与结算

用量事件经双方核对后进入结算。计费使用精确十进制、合同周期、保底及阶梯，账单和费率审批保持独立。

运行需要 Python 3.11 或更高版本，使用标准库与 SQLite，无需第三方包。演练中的机构、人员、交易及读数全部为本地合成数据。

## 构建与测试

```sh
python3 -m compileall -q meterbook
python3 -m unittest discover -s tests -v
```

## 本地运行

```sh
python3 -m meterbook replay fixtures/settlement.json
python3 -m meterbook --help
```

演练使用临时数据库，不会连接远程服务。命令以 JSON 显示业务结果；测试覆盖核心状态、边界输入和错误路径。运行前可用 `python3 --version` 核对解释器版本。
