# VibeSimAgent 重构方案 — 2026-09-10 修订

## 当前实施状态（2026-09-11）

新包、provider/runtime/turn/storage/API、managed launcher兼容和受管理启动迁移已实现；
独立新包测试、真实GPT续聊/委派/SSE/取消/重连及私有迁移演练已有通过证据。
UI主要承接路径和实际发现的设置、引用、分页问题已验证修复，详见
`doc/browser-acceptance.md` 的完成记录；该文档首部矩阵保留的是初始盘点。

当前冻结的是待部署的组件候选版本。尚未完成原计划的Claude历史双角色续聊与生产停写迁移切换、
观察和旧入口/源码退役。生产仍运行旧服务，不能将局部验收或私有迁移演练称为完整完成。
后续按这些有限条件推进，不把UI种类、状态与视口的全部排列另增为完成门槛。

## 决策前提（用户已确认）

1. **数据全部保留**：所有 workspace 复用离线转换器迁移到独立目标；受管理启动
   自动检测支持的旧格式，在确认旧部署停写后、接受请求前执行转换，普通请求处理中不迁移。
   原盘点为 7 个，实际范围以切换时的 registry 和磁盘清单为准。
2. **多 provider 且面向未来可扩展**：镜像同时装 Codex CLI 与 Claude CLI，未来还要加更多。
   provider 必须是一等可扩展轴，不是 `if runner == "claude"`。
3. **无损**：双角色（orchestrated）与单角色（single）都保留，能力零删减。
4. **策略**：新包并行开发，跑通后一次性切换，删除旧 `backend/` 与 `frontend/`。

## 核心诊断

现有结构有三个需要分别解决的问题：provider 与 CLI 耦合、配置加载有导入副作用、
turn 的执行与收尾分散在多个入口。它们不能靠一次目录拆分自动解决。

`CodexFamilySpec` 本质是一个 Codex 概念——它的字段是 `host_codex_home`、
`catalog_filenames`，语义是"Codex 会话恢复兼容边界"。Claude 是硬塞进这个形状的：
`host_codex_home = ~/.claude`、`catalog_filenames = ()`、`available` 里一个
`if self.runner == "claude"` 特判、`agent_cli.run_agent` 一个 17 行字典分发。

新增 provider 需要修改模型配置、容器准备和 turn 的能力特判。
例如 DeepSeek 不启用 CLI 结构化输出的判断就在 `turn.py`，仅替换分发字典不够。

本方案相对现状的主要收益是：新增同类 provider 只增加配置；新增 CLI 只增加 adapter
和供给声明；HTTP 入口共用生命周期；配置和存储可以独立验证。目录数量与净减行数
不是完成标准，也不承诺模型质量或响应速度因重构而提高。

## 目标架构

```
vibesim_agent/
  main.py              仅组装 ASGI app；prompt 渲染在这里显式调用
  settings.py          单一 typed Settings；禁止 import 副作用

  api/                 只做 HTTP，不含业务逻辑
    deps.py            token 鉴权、capability、store 注入
    schemas.py         Pydantic 请求/响应模型
    routers/           workspaces / conversations / files / jobs / catalog / eval

  domain/              纯类型，无 I/O
    events.py          规范化 TurnEvent 词汇表
    roles.py           Role / AgentMode / mode→roles 映射

  services/            编排；无 HTTP、无 SQL
    turn.py            共享生命周期，显式保留 single/orchestrated 的编排差异
    conversation.py    创建/列表/删除/历史
    workspace.py       仓库复制 + git 引导
    naming.py
    evidence.py        analyzer citations

  providers/           ★ 可扩展轴
    base.py            Provider 配置实例、能力模型、CLI adapter 协议
    registry.py        显式注册 provider 实例与 adapter，无导入时自注册
    codex/             command / events / rollout collector（rollout 只存在于此）
    claude/            command / events

  runtime/
    container.py       docker 生命周期，provider 无关
    mounts.py
    process.py         共享的子进程流式读取 + idle timeout + 取消

  storage/
    schema.sql         当前版本完整 schema，运行时检查版本但不执行 ALTER
    registry.py        workspace descriptor（JSON）
    conversations.py   SQLite
    sessions.py        agent_sessions（provider_id + session_scope）
    jobs.py

  prompts/
    render.py          显式渲染步骤
    templates/ contracts/

tools/
  migrate_v1_to_v2.py  离线转换、校验与演练；回滚窗口结束后再归档
```

### Provider 协议要点

区分 provider 配置实例与 CLI adapter。gpt、deepseek 是两个实例，复用 Codex adapter；
claude 实例使用 Claude adapter。通过现有 CLI 接入且能力已覆盖的新模型可由注册解决；
全新协议需要新增 adapter，不承诺未知能力也能零改动支持。

- provider 实例声明身份、凭据来源、模型目录、默认选择与有效能力。
- 能力至少覆盖结构化输出方式、effort、service tier、resume 支持及兼容作用域；
  支持按模型覆盖，非法选择在启动调用前报错。DeepSeek 的差异由能力配置表达。
- adapter 负责命令、MCP 格式、事件解析、会话文件布局、恢复与停止策略；Codex rollout
  读取保留为其私有逻辑。角色决策的校验和 repair/continue 归 turn 服务。
- provider/adapter 声明所需二进制、挂载和环境；runtime 统一校验、组合并执行，
  provider 不直接管理 Docker 生命周期。共享 process 层只抽取确实相同的读写和清理机制。

会话按 workspace、conversation、role 隔离，存储 `provider_id`、`session_scope` 和
`session_id`。作用域必须区分不兼容的 adapter/后端配置，并与实际磁盘会话位置一致；
不能只换一个字符串。兼容模型切换保持 resume，prompt 指纹变化按现有行为不清空会话。
凭据轮换本身不应无条件废弃会话。注册入口可修改，业务编排与存储不按 provider 名分支。

### 事件词汇表统一

两个 runner 已经事实上共享词汇表，只是没写下来，且有一处不一致：
Codex 发 `agent_text`，Claude 发 `intermediate_output`。规范化为单一词汇表：
`role_start / role_ready / session / tool_call / intermediate_output / usage / final`
（角色级）与 `decision / implementer / job / done / error`（turn 级）。

Codex 的 rollout 日志恢复逻辑（`output_collector.py` 371 行）是 Codex 专属的，
收进 `providers/codex/`，不再污染共享路径。两个 CLI 各自重复实现的
idle timeout / stderr tail / 读循环 / 取消上收到 `runtime/process.py`。

词汇表之外，Phase 0 必须冻结每种事件的字段、作用域、顺序与终态语义：

- `role_start` 在启动可能阻塞前发出；`role_ready` 表示 adapter 已确认该轮交接可恢复，
  不能用任意 stdout 字节或心跳代替。每个 adapter 给出对应证据及测试。
- 角色 `final` 不等于整个 turn 完成；仅经过校验的角色结果可触发路由。
  turn 终态区分回答、请求输入、失败、取消，整轮只能有一个持久化终态。
- turn 服务拥有 workspace 串行策略、取消、会话更新、事件记录、收尾与恢复；
  HTTP 层只鉴权、校验和呈现。浏览器断连不取消 turn；同步/eval 的断连政策也要明确测试。
- 明确哪些事件持久化以及发布顺序；保证已记录事件有稳定顺序，重放不重新执行工作。
  旧事件与冻结引用保持原值；内部规范化不能静默改写历史 wire format。
- 保留重复 Stop 只取消一次、指定 turn 的 Stop 不影响后续 turn、排队或启动阶段取消、
  interrupted_role 在释放会话前写入、收尾某一步失败仍执行其余清理等约束。

## 分阶段执行

### Phase 0 — 冻结契约（先做，不改任何生产代码）

先记录准确基线：Agent、UI、Analyzer 的 commit，以及本次继承的未合并改动。
`wt-agent-redesign` 是 UI 配套工作，并非本方案的新包实现；冻结基线要包含已接受的
消息 `id/turn_id`、只读 replay、API 前缀和取消/收尾修复。相关改动先形成可追溯提交，
不在重写过程中继续以移动的工作树作为验收依据。

建立三层验证，旧 backend 先跑通；因已知缺陷新增的用例需先落修复，不能冻结缺陷：

- HTTP 契约测试配合 fake runner，覆盖入口、鉴权、SSE、状态与持久化行为。
- 保留并调整现有 adapter 协议、rollout 恢复、真实子进程管道和取消测试；
  不因测试涉及 CLI 格式就删除它，也不为保留旧 import 路径引入兼容包。
- 切换前执行真实 provider 与容器验收，覆盖首轮、续聊、工具/MCP、混合角色、取消与失败。
  fake runner 全绿不能替代这一层；缺少凭据时明确记录未完成项。

快照完整路由/鉴权清单、事件字段与时序、所有 workspace 的 schema、逐表数量和关联。
路由基准逐项记载采用旧地址还是 `/api/agent/v1`，以及兼容别名的调用方与删除条件。

### Phase 1 — 最小端到端流程 + 配置与存储骨架

- 新包、`settings.py`（见下方「配置体系重整」，消除 `config.py` 导入时读环境/写磁盘的副作用）
- 当前版本完整 `schema.sql`；启动前检测全状态版本，已支持的旧版本由受管理启动流程调用
  离线转换器自动升级，未知、混合或损坏版本明确拒绝。
- 实现最小 provider/adapter 接口，先打通 single + Codex 的 HTTP → turn → adapter →
  SQLite → 历史/续聊流程，同时提供 fake adapter 供快速测试，不等 Phase 4 才接 API。
- prompt 在显式启动/构建函数中准备；uvicorn、CLI 与测试均有明确入口，不靠 import 写文件。
- 新入口提供init/serve/env-reference：init只接受不存在的状态根目录，创建external w_main
  与当前schema；serve对含归档的全部DB预检，不隐式初始化。自动迁移在受管理启动协调阶段完成，
  不在请求处理或factory内部进行。factory在写共享prompts前
  取得状态目录独占锁，再交给lifespan恢复与shutdown释放，拒绝prompts目录及文件符号链接。
- 验收：这条流程的首轮、续聊、断连重连、取消和失败收尾通过，测试使用隔离数据目录。

### Phase 2 — Provider 插件层

- `providers/base.py` 协议 + `runtime/process.py` 共享流式循环
- 完成 Codex 与 Claude adapter、能力配置和容器供给；复用同一端到端流程验收。
- 验收：新注册 dummy provider 可被 catalog 列出并跑完一轮，业务/存储无新增分支；
  GPT/DeepSeek 共用 adapter 且结构化输出策略不同；各 adapter 的解析、超时、启动取消、
  进程退出和 durable result 测试通过。

### Phase 3 — Turn 服务

- 完成 orchestrated 与 single 的共享生命周期，保留模式各自的角色编排
- prompt 由显式构建步骤渲染，4 份 AGENTS 契约保持不变
- 启动恢复在接收请求前执行，并持有整个 workspace 状态根目录的进程独占锁，
  直到关闭服务并等待活动任务结束。检查包括归档工作区在内的所有数据库；
  对遗留 running turn 先撤销 capability、确认所属容器已移除，再删除旧 context，
  再原子写入 interrupted 状态和重启说明。保留角色 home/session、恢复角色与原始事件字节，
  不补造正常 final/done；容器清理或数据库失败则启动失败，保留可重试状态。
  该锁协调新后端实例；旧服务不使用此锁，切换前仍必须停写并停止旧服务。
- 验收：含双角色、混合 provider、repair/continue/durable handoff、定向恢复、并发 Stop、
  workspace 排队取消、收尾异常和服务重启恢复；SSE、同步和 eval 对同一终态解释一致。

### Phase 4 — 完整 API 与迁移演练

- 拆 router；三个入口（浏览器 SSE / agent 同步 / eval）统一到同一个 turn service
  ——SSE 与同步是同一条事件流的两种**呈现**，不是两套实现
- 补齐文件、作业、证据、命名等路由；验收冻结基线的 VibeSimUI，不额外改变已冻结的接口。
- 实现并在隔离快照上演练迁移，满足下方全部数据与恢复标准。

#### 数据迁移与恢复标准

- 输入包含 registry、workspace descriptor、SQLite 及 WAL 的一致性快照、每角色会话目录、
  repo/logs/artifact 路径和作业关联；先 dry-run 给出源/目标清单，再写独立目标，失败可重跑。
- 保留 workspace/conversation/message/turn/job/resource ID、消息时间、事件 sequence、
  引用字节和关联关系；显式保留 messages 主键与 AUTOINCREMENT 高水位。历史 `turn_id`
  为空时继续为空。字段更名/枚举转换须列映射，不能用行数一致代替逐项验证。
- 保留角色 session 文件及兼容映射；容器重建后的挂载目标能找到旧会话。
  `w_main` 只迁移描述与状态，不复制或改写其外部源码、日志与仓库。
- 校验逐表内容、引用和文件完整性，并在快照副本上真实续聊已有 Codex/Claude 会话，
  包含双角色；不能只证明历史可读。演练副本使用独立容器和可写 repo，不挂载真实
  `w_main` 作写入验收目标。
- 演练报告记录输入版本、转换映射、校验结果和恢复结果；生产迁移在 Phase 6 停写后执行。
- 已完成GPT子集真实验收：完整私有源迁移保留5个历史session（含失败Claude），GPT双角色
  各两轮保持原session ID、精确召回旧marker并推进原transcript。两次应用lifespan及容器重建
  在同Python进程内完成；旧state/repo不变，测试容器已清理。Claude和完整turn编排仍待验收，
  子集结果不替代默认四角色要求。
- GPT完整任务编排已通过真实provider/container与HTTP ASGI验收：orchestrator单次委派，
  implementer写入唯一测试文件，orchestrator沿用原session收尾；唯一done、history、SSE与replay
  一致，私有repo最终仅测试文件和两份角色规定的会话记录变化。此前两次失败记录保留。
  此验收不覆盖真实网络渐进SSE、取消或Claude编排，生产未切换。
- 后续真实Uvicorn/TCP渐进SSE与GPT取消通过：10帧到达后观察唯一tag等待进程，再定向取消；
  容器删除前确认进程退出，唯一cancelled终态、session保留、history/replay、重复cancel一致。
  服务/容器/端口均清理，旧state/repo不变。真实断连重连与Claude仍待验收。
- 真实GPT断连重连随后通过：关闭消息流连接后同进程及turn继续运行，GET stream恢复原8帧，
  不重新POST消息，最终唯一取消终态与history/replay/session一致。资源已清理，独立复审通过。
  Claude三次原始503均明确No available accounts，待网关恢复，不以更换模型规避原验收要求。

#### 启动时自动迁移（用户已确认）

- 已完成迁移的状态直接启动；首次检测到支持的旧格式，在接受任何请求前自动完成迁移。
  全部workspace（包括归档）必须使用同一支持格式，不能只检查w_main或信任版本号。
- 由明确管理旧部署的启动协调器停止其接收任务、排空或取消活动任务，停止旧服务及其
  自动重启入口，等待旧进程和所属容器退出。状态树内写日志的服务也需停止或迁出日志。
  旧backend没有参与新版目录锁；空闲端口、锁获取成功或重复文件扫描一致不能充当停写证明。
  无法确认部署归属时阻止首次迁移并给出具体原因，不猜测或停止其他用户服务。
- 复用离线转换器，使用最终provider配置生成明确映射，在固定独立目标保留原数据并校验。
  转换成功后原子发布独立的源到目标选择记录；重启使用该目标，不重复迁移旧源。
  记录与源身份、目标、映射和完整成功manifest绑定，拒绝已有的不完整或不匹配目标。
- Agent和Analyzer统一使用已验证目标。迁移或启动失败不覆盖原数据；恢复用户写入后按
  Phase6规则处理新增数据，不自动退回旧快照。真实provider续聊验收仍不可省略。
- 当前进度：只读版本检测已实现并通过10项专项及独立复审；目标选择记录组件已通过12项专项。
  prepare_startup协调逻辑已实现，新增10项专项，三组件组合32项通过且独立复审通过；
  专用tmux停写adapter已实现，12项专项与独立复审通过；真实隔离tmux/Docker停写接自动迁移、
  协调器第二次调用保留新版数据均通过，旁路资源未误停、测试资源已清理；
  停机完成凭据/重启阶段已实现，19项专项与独立复审通过。
  serve --startup-config及只读selected-root入口已实现，12项专项、80项组合与独立复审通过；
  后续已补迁移后runtime scope重验，14项专项1.378s通过且独立复审通过；
  真实新app两轮启动/正常关闭/重启验收已通过，HTTP会话与工作区名称持久化、选择记录与旧source
  不变，selected-root一致；未调用provider或启动Analyzer进程，部署脚本切换仍待验收。
  后续Analyzer进程联动发现现有binary仍使用旧API前缀，已独立构建匹配源码的binary并重试通过：
  同一目标的目录发现、动态归档/恢复及ID保留均正确，旧source/selection/receipt不变，
  两进程正常关闭且端口释放。这是合成目录资源联动验收，不覆盖数值分析或provider续聊；
  候选部署脚本已共享startup-config、使用独立日志及显式Analyzer binary，复审通过但未执行。
  停写context退出不能自动恢复旧backend，异常不等于允许回滚。

### Phase 5 — managed job 端点统一（跨仓库，单列）

`/api/internal/managed-runs/*` 与 `/api/internal/managed-jobs/*` 的重复**跨两个仓库**：
调用方是 `VibeSim/launcher/managed_run.py` 和 `managed_job.py`，后者自己还带着
`LEGACY_MANAGED_RUN_CONTEXT_ENV`。而容器里烘焙的是某个版本的 VibeSim checkout，
存在版本偏斜。

因此：新端点 `job_kind: simulation | timing_predict | kernel_profile | kernel_measure`
统一，**仍被调用的旧路径保留为薄别名**，包括前缀迁移前的路径，同时提 VibeSim 侧改动。
镜像重建不会更新已有 workspace 中的 launcher；盘点并验证这些副本、工具和启动脚本，
所有调用方迁移后再单独删除别名。未迁移的历史副本继续得到兼容服务。

### Phase 6 — 停写切换、观察与删除

- 先通过分层验收和迁移演练，准备旧代码/镜像/配置与完整状态备份及恢复步骤。
- 停止接收新 turn，排空或明确取消活动 turn 和 managed job，确认容器内写入进程已退出、
  回调已停止后制作最终一致性快照。不得让新旧服务同时写同一份状态。
- 执行迁移与校验，再切换服务、配置和容器；在恢复用户写入前完成读操作与隔离会话冒烟。
  此时失败可恢复旧服务和原状态；恢复用户写入后不能直接回滚旧快照，必须先保全并处理
  新增数据，选择经验证的反向转换或向前修复。切换手册写明负责人和回滚窗口。

- 删除 `backend/`、`frontend/`、`frontend/tools/shots/`、`migrate_workspaces.py`
- `run.sh` 不再 npm ci/build；`app.py` 的 `GET /` 与 StaticFiles 挂载移除
- `pyproject.toml` 更名（`vibesim-user-facing-ui` → `vibesim-agent`），logger
  `vibesim_ui.*` → `vibesim_agent.*`，`UI_DIR` → `REPO_ROOT`
- Docker 镜像与 `CODEX_*` 环境变量更名为 `VIBESIM_RUNNER_*`（见「配置体系重整」）；
  改名即触发镜像重建，与重建合并成一次
- 修掉写死的个人用户名 fallback：`scripts/build-codex-runner-image.sh` 的
  `app_user="${CODEX_DOCKER_USER:-${USER:-kanzhu}}"` 与
  `docker/codex-runner.Dockerfile:13` 的 `ARG APP_USER=kanzhu`。
  （用户已确认留到本阶段，随镜像重建一起做，不单独提前修。）
- 同批更新携带旧变量名的运行态：`agent-workspaces/services/` 五个启动脚本、
  tmux `vibesim-kanzhu` 会话；已有容器需重建才能拿到新的注入集合
- 重写 README（42KB，通篇仍是 "chat UI" 框架）与 SKILL.md
- 旧源码可在切换提交中删除，旧部署包和数据备份保留至回滚窗口结束；迁移工具随后归档。
  删除独立 frontend 的前提是冻结的浏览器能力已由 VibeSimUI 全部承接。

## 配置体系重整（环境变量）

### 现状：60+ 个变量名，README 记录约 40 个，至少 9 个无文档

以下数量来自初次审计，Phase 0 重新盘点。治理重点是明确配置归属、加载时机与校验，
而非单纯减少变量数；配置问题与 provider 耦合有关，但需要独立完成标准。

**缺陷 1 — `CODEX_` 前缀承载六种互不相干的含义**，真正指 Codex CLI 的只有
`CODEX_NPM_PACKAGE` 一个：

| 实际含义 | 变量 |
| --- | --- |
| gpt 家族模型默认值 | `CODEX_MODEL`、`CODEX_REASONING_EFFORT`、`CODEX_TRADITIONAL_HOME` |
| 容器身份/配置（与 Codex 无关） | `CODEX_DOCKER_*` 共 9 个 |
| 镜像构建（provider 无关） | `CODEX_CUDA_IMAGE`、`CODEX_UV_IMAGE`、`CODEX_NPM_PACKAGE`、`CODEX_RUNNER_IMAGE_VERSION`、`CODEX_{SKIP,FORCE}_IMAGE_BUILD`、`CODEX_SKIP_RUNNER_IMAGE_TEST` 共 7 个 |
| 所有 runner 通用的超时 | `CODEX_IDLE_TIMEOUT`（`claude_cli.py` 也 import） |
| VibeSim 的 uv.lock 哈希 | `CODEX_MAIN_LOCK_SHA` |

**缺陷 2 — 一个 provider 四个名字。** gpt 家族 = `gpt`(family_id) +
`CODEX_MODEL`(env) + `CODEX_TRADITIONAL_HOME`(env) + `traditional`(legacy key)。
deepseek = `deepseek` + `CODEXDS_*` + `codexds`。

**缺陷 3 — 三个 provider 三套方案，且有洞：**

| | model | effort | home |
| --- | --- | --- | --- |
| gpt | `CODEX_MODEL` | `CODEX_REASONING_EFFORT` | `CODEX_TRADITIONAL_HOME` |
| deepseek | `CODEXDS_MODEL` | `CODEXDS_REASONING_EFFORT` | `CODEXDS_HOME` |
| claude | `CLAUDE_MODEL` | 缺失 | 硬编码 `~/.claude` |

加第四个 provider 就要发明第四套方案。

**缺陷 4 — 同名跨作用域两义。** `CODEX_DOCKER_GPUS`、`ANALYZER_MCP_SOURCE`、
`ANALYZER_MCP_BASE_URL` 在宿主机读作"默认值"，又以完全相同的名字注入容器作"实际值"。
`HF_HOME` 宿主机是源路径、容器里是 `/model`。

**缺陷 5 — 五种作用域拍平进同一命名空间**：宿主进程配置 / provider 凭据 /
模型默认值 / 镜像 build arg / 容器注入 env / launcher capability 交接。

**缺陷 6 — 构建脚本自身不一致。** 读 `CODEX_CUDA_IMAGE` 传成 build-arg `CUDA_IMAGE`；
`NODE_VERSION`/`NODE_ARCH`/`RUST_TOOLCHAIN` 零前缀占用全局名。
附带真实缺陷：`app_user="${CODEX_DOCKER_USER:-${USER:-kanzhu}}"` 与 Dockerfile
`ARG APP_USER=kanzhu` 把个人用户名写死成 fallback。

**缺陷 7 — 兼容别名层叠**：`CODEX_IDLE_TIMEOUT` ← `CODEX_TURN_TIMEOUT`；
`OPENROUTER_API_KEY` ← `OPENROUTE_KEY`；`VIBESIM_MANAGED_JOB_CONTEXT` 与
`VIBESIM_MANAGED_RUN_CONTEXT` 同时写入且指向同一文件。

**缺陷 8 — 无文档**：`VLLM_API_KEY`、`CODEX_MAIN_LOCK_SHA`、
`VIBESIM_EXPECTED_LOCK_SHA`、`ANALYZER_MCP_ANALYZE_BIN`、
`MAIN_TREE_SKIPPED_SUBMODULES`、`VIBESIM_BASE_URL`、`SMOKE_AGENT_MODE`、
`NODE_VERSION`、`NODE_ARCH`。

**缺陷 9 — 约 20 个在 `config.py` 模块导入时读取**，无校验、无法按测试覆盖。
`int(os.environ.get("CODEX_DOCKER_UID", ...))` 遇坏值抛裸 ValueError。

### 治理规则

**规则 1：作用域即前缀第二段。** 一个产品前缀，作用域显式区分。

```
VIBESIM_AGENT_*           宿主后端进程（bind、token、workspaces root、超时）
VIBESIM_RUNNER_*          容器运行时与镜像构建
VIBESIM_PROVIDER_<ID>_*   每个 provider，格式统一
```

**规则 2：provider 变量前缀从稳定 id 派生，只暴露有实际用途的字段。**
注册时校验 id 字符集及大小写归一化后的唯一性。例如：

```
VIBESIM_PROVIDER_GPT_MODEL / _EFFORT / _HOME
VIBESIM_PROVIDER_DEEPSEEK_MODEL / _EFFORT / _HOME
VIBESIM_PROVIDER_CLAUDE_MODEL / _EFFORT
```

`_HOME` 只用于确实从宿主目录读取配置的 provider，须明确它是凭据来源还是会话存储。
Claude 当前由环境凭据认证，不能为对齐 GPT 而复制宿主 Claude 历史或添加无效 `_HOME`。
effort/tier 默认值只在模型支持时生效；支持范围由能力模型验证。

**规则 3：第三方凭据保留上游拼写。** `ANTHROPIC_API_KEY`、`OPENROUTER_API_KEY`、
`VLLM_API_KEY`、`HF_HOME` 是别人的契约，不改名。provider 声明所需凭据；HF 模型缓存
属于 runtime 挂载配置。统一加载入口解析这些声明，业务编排不直接读环境，日志不输出秘密。

**规则 4：配置按所有者分组，派生的容器注入值不再作为第二套配置来源。**

```
AgentSettings       宿主进程
ImageSettings       build arg —— 只有构建脚本读
ContainerSettings   runtime 配置；由它与 provider 声明计算 docker run 注入内容
ProviderSettings    按 provider id 派生
ExternalSecrets     由 provider 声明，核心不读
```

容器注入集合由**一处显式构造**，取代 docker.py 里 13 个内联 `-e` 字面量，
使"容器到底能看见什么环境"成为一个可读函数。

**规则 5：禁止 import 时读环境。** 单一 `load_settings()` 由 `main.py` 调用，
做类型与取值校验并给出清晰错误。业务测试直接构造 Settings；配置加载测试仍需覆盖
环境输入、默认值、非法值和秘密脱敏。离线工具显式调用所需配置入口。

**规则 6：自有配置区分宿主源值与容器目标值。** 第三方变量在容器里保留上游名字，
例如宿主 HF 缓存映射到容器 `HF_HOME=/model`；显式记录映射，不为改名破坏第三方契约。

**规则 7：README 的变量表由 settings 定义生成**，不手工维护——40 个手写条目
已经漏了 9 个，手工维护必然继续腐化。

**规则 8：切换时删除已完成调用方迁移的别名。** `CODEX_TURN_TIMEOUT`、`OPENROUTE_KEY`
在启动配置迁移后删除；部署检查发现已废弃配置时明确报错，避免静默退回错误默认值。
`VIBESIM_MANAGED_{RUN,JOB}_CONTEXT` 及旧回调地址按 Phase 5 的实际调用方清单处理。

### 落点

规则 1–7 在 Phase 1 随 `settings.py` 落地；`VIBESIM_RUNNER_*` 的构建脚本与
Dockerfile 改名在 Phase 6 与镜像重建一起做（改名即触发重建，合并成一次）；
规则 8 的跨仓库部分在 Phase 5。

## 必须无损保留的能力清单（验收面）

1. 两种 agent_mode × autonomous = 4 份 AGENTS 契约
2. 三个角色：orchestrator / implementer / assistant
3. 三个 provider：gpt、deepseek、claude，且可扩展
4. per-role 的 model / effort / service_tier 选择与 resume 兼容性锁定
5. 浏览器 SSE 的重连重放、取消、`interrupted_role` 中途转向
6. token 鉴权同步接口与公开 skill 文档；地址以 Phase 0 冻结路由表为准
7. eval 单轮接口；地址以 Phase 0 冻结路由表为准
8. managed 回调：simulation run + 三种 typed job
9. 五类 analyzer citation 字典 + freezing + MCP 注册
10. analyzer evidence MCP server 注入容器
11. workspace 复制（VibeSim tracked 文件 + git init）；w_main 外部直通
12. workspace 自动命名
13. 文件预览/列表/meta + artifact 下载
14. 容器生命周期：per-conversation、GPU、HF 模型挂载、submodule 挂载、prompt 挂载、`--init`
15. 传输失败分类
16. idle timeout 与取消（向容器内发信号）
17. durable final handoff 恢复
18. 已接受的消息 id/turn_id、只读 turn replay、X-Turn-Id 与定向取消契约

## 风险

- **既有冷缓存 profiling 部署缺口**：真实新 runner 在 B200 上完成本地
  elementwise、RMSNorm 与 GEMM profiling 后，因 `kv_cache_append:vllm_cuda`
  使用 `ContainerProfileEnv` 而报 `docker is required for container profiling`。
  旧 runner 同样没有 Docker CLI 或宿主 socket 挂载；这不是变量更名引入的回归。
  本计划保留 GPU 容器能力，不新增远程 profiling 服务。跨容器 profiling 的执行、
  宿主路径与 GPU 映射应作为独立后续解决，不能通过挂载宿主 socket 冒充镜像小修。
  完整冷缓存 timing 仍未通过，部分 kernel 成功与暖缓存运行均不能替代该验收。
- **最大风险是只测 fake runner 就宣称无损**。HTTP 契约、adapter/真实子进程测试、
  真实 provider 与旧会话续聊验收各自覆盖不同风险，不能互相替代。
- **容器镜像与 VibeSim 版本偏斜**：Phase 5 的跨仓库改动需要重建镜像，
  已有 workspace 副本里的 launcher 代码是旧的。
- **w_main 是外部直通**（指向真实 VibeSim checkout），迁移脚本要特判，
  不能当成 managed workspace 复制。
- **环境变量改名会打到运行中的服务**：`agent-workspaces/services/*.sh` 五个本地启动
  脚本、tmux `vibesim-kanzhu` 会话、以及已有容器都携带旧变量名。切换时这些必须同批更新，
  且旧容器需要重建才能拿到新的注入集合。

## 完成标准与规模预期

相比当前实现，应能用以下结果证明改善：新增同 adapter 的 provider 无业务分支；
新增 CLI 不修改 turn/store；三个入口共用终态、取消与收尾规则；import 不读环境或写磁盘；
原有功能与会话通过分层验收。只按实际职责拆文件，不新增通用插件框架或空转发层。

初始粗估：`backend/` 11,264 行 → 约 9,000 行（净减约 2,300：迁移工具 378、store 遗留 ~200、
managed 端点去重 ~150、双 CLI 循环去重 ~150、三入口去重 ~200，其余为分层拆分不减量）。
`frontend/` 2,540 行整体删除。这些数字尚未按最新 worktree 重算，不作为验收指标。
