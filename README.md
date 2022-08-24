# **青训营结业项目答辩汇报**

## 一、项目介绍
- 项目概括：实现一个简易分布式计算系统最核心的部分：MySQL协议、SQL解析、验证、优化器、执行器、多种支持算子等等，在这里我们不考虑垃圾回收、高并发安全等拓展功能。

- 项目服务地址:
- Github地址：https://github.com/bytedance-training-camp/simple-distributed-sql-engine
## 二、项目分工











## 三、项目实现

### 3.1 技术选型与相关开发文档

















### 3.2架构设计

![image](https://github.com/bytedance-training-camp/Youth-training-camp-defense-document/blob/main/img/%E6%9C%AA%E5%91%BD%E5%90%8D%E7%BB%98%E5%9B%BE.drawio%20(1)(1).png)

- 首先，输入命令后，进入server层，这里是与用户交互的地点，目前tinysql支持 MySQL协议。功能是解析MySQL命令并返回执行结果，具体的实现按照MySQL协议实现。

- 之后，SQL语句会经过parser模块进行语法解析，最终得到AST后进入Optimizer模块，优化查询计划
- 然后根据计划生成查询器->执行并返回结果

以上主要内容对应于以下的包中：

| server   | MySQL协议                                     |
| -------- | --------------------------------------------- |
| parser   | 语法验证                                      |
| planner  | 合法性验证+指定查询计划+优化查询计划          |
| executor | 执行器生成以及执行                            |
| kv       | kv层提供存储需求                              |
| dissql   | 通过 TiKV Client 向 TiKV 发送以及汇总返回结果 |


### 3.3项目代码介绍
#### 用户界面-网络协议

关于如何实现MySQL协议的代码都放在server包中。启动server时，首先在协议层入口，有一个Go retinue监听端口， for 循环起两个 Listener goroutine 处理客户端发过来的消息。等待从客户端发来的包，并对发来的包做处理

```go
//启动server
func (s *Server) Run() error {
	go s.startNetworkListener(s.listener, false, errChan)
	go s.startNetworkListener(s.socket, true, errChan)
        ......
}

func (s *Server) startNetworkListener(listener net.Listener, isUnixSocket bool, errChan chan error) {
	for {
		conn, err := listener.Accept()
        ......
		go s.onConn(clientConn)
	    }
}
```



在server文件夹的conn中，这里可以认为是分布式存储系统的入口。首先在clientConn Run( )中，这里会在一个循环中，不断读取网络包。

```go
data, err := cc.readPacket()//在一个循环中，不断的读取网络包
```

然后调用dispatch()方法处理收到的请求：

```go
if err = cc.dispatch(ctx, data); err != nil {
```

接下来就进入clientConn.dispatch()方法：

```go
func (cc *clientConn) dispatch(ctx context.Context, data []byte) error {
```

此处要处理的包是原始byte数组，第一个byte即为command的类型

```go
cmd := data[0]
```

然后我们拿到command的类型后，根据类型调用对应的处理函数，最常用的Command是COM_QUERY，对于大部分SQL语句，只要不是用prepared方式，都是COM_QUERY。对于COM_QUERY，从客户端发来的主要是SQL文本，处理函数是handleQuery()，这个函数会调用具体的执行逻辑：	

```go
func (cc *clientConn) handleQuery(ctx context.Context, sql string) (err error) {
    rss, err := cc.ctx.Execute(ctx, sql)
```

其中用到的Execute方法在driver_tidb.go中

```go
func (tc *TiDBContext) Execute(ctx context.Context, sql string) (rs []ResultSet, err error) {
rsList, err := tc.session.Execute(ctx, sql)
```

其中session Execute的实现在session.go中，自此会进入SQL核心层。经过一系列处理后，拿到SQL语句的结果会调用 writeResultset方法把结果写回客户端

```go
err = cc.writeResultset(ctx, rss[0], false, 0, 0)
```

以上是协议层的入口。协议层的出口用到上面提到的writeresult方法，根据MySQL协议的要求，把结果写回客户端。
![image](https://github.com/bytedance-training-camp/Youth-training-camp-defense-document/blob/main/img/%E5%9B%BE%E7%89%871.png)

------


#### SQL解析与验证
![image](https://github.com/bytedance-training-camp/Youth-training-camp-defense-document/blob/main/img/QQ%E5%9B%BE%E7%89%8720220821094807.jpg)

- **parser**

Parser是将输入文本转换为AST（抽象语法树），parser有包括两个部分，Parser和Lexer，其中Lexer实现词法分析，Parser实现语法分析。

- **AST**

AST是abstract syntax tree的缩写，也就是抽象语法树。和所有的Parser一样，Druid Parser会生成一个抽象语法树。

- **Lexer&Yacc**

Lexer:Lex会生成一个叫做『词法分析器』的程序。这是一个函数，它带有一个字符流传入参数，词法分析器函数看到一组字符就会去匹配一个关键字(key)，采取相应措施。

```c++
{
#include <stdio.h>
}
stop printf("Stop command received\n");
start printf("Start command received\n");
```

编译需执行以下命令：

```go
lex example1.l
cc lex.yy –o example –ll
```

Yacc:用来为编译器解析输入数据，即程序代码。还可以解析输入流中的标识符(token)。

==这两个组件共同构成了 Parser 模块，调用 Parser，可以将文本解析成结构化数据，也就是AST（抽象语法树）==

```go
session.go 699:     return s.parser.Parse(sql, charset, collation)
```

```go
session.go 699:     return s.parser.Parse(sql, charset, collation)
```

在解析过程中，会先用 [lexer ](https://github.com/pingcap/tidb/blob/source-code/parser/lexer.go)不断地将文本转换成 token，交付给 Parser，Parser 是根据 [yacc 语法 ](https://github.com/pingcap/tidb/blob/source-code/parser/parser.y)生成，根据语法不断的决定 Lexer 中发来的 token 序列可以匹配哪条语法规则，最终输出结构化的节点。 例如对于这样一条语句 `SELECT * FROM t WHERE c > 1;`，可以匹配 [SelectStmt 的规则 ](https://github.com/pingcap/tidb/blob/source-code/parser/parser.y#L3936)，被转换成下面这样一个数据结构：

```go
 type SelectStmt struct {
        dmlNode
        resultSetNode
    
        // SelectStmtOpts wraps around select hints and switches.
        *SelectStmtOpts
        // Distinct represents whether the select has distinct option.
        Distinct bool
        // From is the from clause of the query.
        From *TableRefsClause
        // Where is the where clause in select statement.
        Where ExprNode
        // Fields is the select expression list.
        Fields *FieldList
        // GroupBy is the group by expression list.
        GroupBy *GroupByClause
        // Having is the having condition.
        Having *HavingClause
        // OrderBy is the ordering expression list.
        OrderBy *OrderByClause
        // Limit is the limit clause.
        Limit *Limit
        // LockTp is the lock type
        LockTp SelectLockType
        // TableHints represents the level Optimizer Hint
        TableHints []*TableOptimizerHint
    }
```

大部分 ast 包中的数据结构，都实现了 `ast.Node`接口，这个接口有一个 `Accept`方法，后续对 AST 的处理，主要依赖 Accept 方法，以 [Visitor 模式 ](https://en.wikipedia.org/wiki/Visitor_pattern)遍历所有的节点以及对 AST 做结构转换。

- **制定查询计划及其优化**

得到AST之后，就可以对其进行各种验证、变化、以及优化，可通过如下语句进行操作：

```go
session.go 805:             stmt, err := compiler.Compile(goCtx, stmtNode)
```

进入Compile函数后，有三个重要的步骤：

1. `plan.Preprocess`: 做一些合法性检查以及名字绑定；
2. `plan.Optimize`：制定查询计划，并优化，这个是最核心的步骤之一，后面的文章会重点介绍；
3. 构造 `executor.ExecStmt`结构：这个 [ExecStmt ](https://github.com/pingcap/tidb/blob/source-code/executor/adapter.go#L148)结构持有查询计划，是后续执行的基础，非常重要，特别是 Exec 这个方法。

- **生成执行器**

1. 首先我们要提取出执行器的接口，定义出执行方法、事务获取和相应提交、回滚、关闭的定义，同时由于执行器是一种标准的执行过程，所以可以由抽象类进行实现，对过程内容进行模板模式的过程包装。在包装过程中定义抽象类，由具体的子类来实现。
2. 之后是对 SQL 的处理，在执行 SQL 的时候，分为了简单处理和预处理，预处理中包括准备语句、参数化传递、执行查询，以及最后的结果封装和返回。
![image](https://github.com/bytedance-training-camp/Youth-training-camp-defense-document/blob/main/img/1_c3e07627e9.png)

具体代码：

```go
executor/adpter.go 227:  e, err := a.buildExecutor(ctx)
```

生成执行器之后，[封装在一个 `recordSet`结构中 ](https://github.com/pingcap/tidb/blob/source-code/executor/adapter.go#L260)：

```go
return &recordSet{
            executor:    e,
            stmt:        a,
            processinfo: pi,
            txnStartTS:  ctx.Txn().StartTS(),
        }, nil
```

- **运行执行器**

TiDB 的执行引擎是以 Volcano 模型运行，所有的物理 Executor 构成一个树状结构，每一层通过调用下一层的 Next/NextChunk() 方法获取结果。

![执行器树](https://github.com/bytedance-training-camp/Youth-training-camp-defense-document/blob/main/img/image-20220818150800016.png)

这里的 `rs`即为一个 `RecordSet`接口，对其不断的调用 `Next()`，拿到更多结果，返回给 MySQL Client。 第二类语句是 Insert 这种不需要返回数据的语句，只需要把语句执行完成即可。这类语句也是通过 `Next`驱动执行，驱动点在 [构造 `recordSet`结构之前 ](https://github.com/pingcap/tidb/blob/source-code/executor/adapter.go#L251)：

```go
// If the executor doesn't return any result to the client, we execute it without delay.
        if e.Schema().Len() == 0 {
            return a.handleNoDelayExecutor(goCtx, e, ctx, pi)
        } else if proj, ok := e.(*ProjectionExec); ok && proj.calculateNoDelay {
            // Currently this is only for the "DO" statement. Take "DO 1, @a=2;" as an example:
            // the Projection has two expressions and two columns in the schema, but we should
            // not return the result of the two expressions.
            return a.handleNoDelayExecutor(goCtx, e, ctx, pi)
        }
```


- **用Druid SQL Parser解析SQL**

Druid SQL Parser分三个模块：Parser，AST，Visitor。

在Druid Parser中可以通过如下方式生成AST：

```java
final String dbType = JdbcConstants.MYSQL; // 可以是ORACLE、POSTGRESQL、SQLSERVER、ODPS等
String sql = "select * from t";
// SQLStatement就是AST
List<SQLStatement> stmtList = SQLUtils.parseStatements(sql, dbType);
```

在使用过程中，需加入依赖：

```go
<dependency>
    <groupId>com.alibaba</groupId>
    <artifactId>druid</artifactId>
    <version>1.2.6</version>
    <scope>test</scope>
</dependency>
```

- **Visitor**

Visitor是遍历AST的手段，是处理AST最方便的模式，Visitor是一个接口，有缺省什么都没做的实现VistorAdapter。

Druid内置提供了如下Visitor:

OutputVisitor用来把AST输出为字符串
WallVisitor 来分析SQL语意来防御SQL注入攻击
ParameterizedOutputVisitor用来合并未参数化的SQL进行统计
EvalVisitor 用来对SQL表达式求值
ExportParameterVisitor用来提取SQL中的变量参数
SchemaStatVisitor 用来统计SQL中使用的表、字段、过滤条件、排序表达式、分组表达式
SQL格式化 Druid内置了基于语义的SQL格式化功能


- **SQL Compile**

SQL Compiler 处理层完成 SQL 语意检查(Preprocess)、编译执行计划(Logical Optimize、Physical Optimize) 工作，将 SQL 编译成可执行的物理执行计划。

   1、 首先，从 MySQL Protocol Layer 串联起 “解析”、“执行” 操作，并在 [func (s *session) ExecuteStmt(…)](https://github.com/pingcap/tidb/blob/27348d67951c5d9e409c84ca095f0e5d3332c1fd/session/session.go#L1735) 中调用  [func (c *Compiler) Compile(…)](https://github.com/pingcap/tidb/blob/27348d67951c5d9e409c84ca095f0e5d3332c1fd/executor/compiler.go#L51-L109)  进行真正的编译处理。

   2、其次，在 Compile 内部调用 [func Preprocess(…)](https://github.com/pingcap/tidb/blob/27348d67951c5d9e409c84ca095f0e5d3332c1fd/planner/core/preprocess.go#L114-L130)，进行 Preprocess 完成前置检查，如：语义检查。具体实现流程为，通过 AST 的  [Accept 方法](https://github.com/pingcap/tidb/blob/27348d67951c5d9e409c84ca095f0e5d3332c1fd/parser/ast/ast.go#L40) 方法, 构造一个 Vistor 实现对 AST 的遍历。 每个 Visitor 接口包含 Enter、Leave 方法，[并在 Enter 或 Leave 时](https://github.com/pingcap/tidb/blob/27348d67951c5d9e409c84ca095f0e5d3332c1fd/planner/core/preprocess.go#L192)，依据 SQL 类型进行判断。本例 point get 会跳到  [func (n *SelectStmt) Accept(v Visitor)](https://github.com/pingcap/tidb/blob/27348d67951c5d9e409c84ca095f0e5d3332c1fd/parser/ast/dml.go#L1391-L1503) 中，不断分支处理完成遍历。
   3、最后，进入 Optimizer 处理，本例中因为是点查会越过大量优化器处理过程，直接进入 [func TryFastPlan(…)](https://github.com/pingcap/tidb/blob/27348d67951c5d9e409c84ca095f0e5d3332c1fd/planner/optimize.go#L131) 进行简单的 “权限检查” 及 “数据库名检查”。

```go
// TryFastPlan tries to use the PointGetPlan for the query.
func TryFastPlan(ctx sessionctx.Context, node ast.Node) (p Plan) {
    ......
	case *ast.SelectStmt:
		if fp := tryPointGetPlan(ctx, x, isForUpdateReadSelectLock(x.LockInfo)); fp != nil {
			if checkFastPlanPrivilege(ctx, fp.dbName, fp.TblInfo.Name.L, mysql.SelectPriv) != nil {
				return nil
			}
			if tidbutil.IsMemDB(fp.dbName) {
				return nil
			}
			if fp.IsTableDual {
				return
			}
			p = fp
			return
		}
	}
	return nil
}
```
------
#### 优化部分

此模块负责进行合法性检查及名字绑定、由AST生成逻辑执行计划，并基于一系列优化规则进行优化，生成物理执行计划并返回。

模块的入口为：

```go
// executor/compiler.go
// 将AST结点转化为物理执行计划
func (c *Compiler) Compile(ctx context.Context, stmtNode ast.StmtNode) (*ExecStmt, error) {
    // 进行合法性检查及名字绑定
    infoSchema := infoschema.GetInfoSchema(c.Ctx)
    if err := plannercore.Preprocess(c.Ctx, stmtNode, infoSchema); err != nil {
        return nil, err
    }

    // 优化，生成物理执行计划
    finalPlan, names, err := planner.Optimize(ctx, c.Ctx, stmtNode, infoSchema)
    if err != nil {
        return nil, err
    }

    return &ExecStmt{
        InfoSchema:  infoSchema,
        Plan:        finalPlan,
        Text:        stmtNode.Text(),
        StmtNode:    stmtNode,
        Ctx:         c.Ctx,
        OutputNames: names,
    }, nil
}
```

在`Optimize`函数中，首先由AST构造逻辑执行计划：

```go
// planner/optimize.go
builder := plannercore.NewPlaBuilder(sctx, is)
p, err := builder.Build(ctx, node)
if err != nil {
    return nil, nil, err
}
logic, isLogicalPlan := p.(plannercore.LogicalPlan)
```

再进行优化，生成最终执行计划：

```go
// planner/optimize.go
finalPlan, err := plannercore.DoOptimize(ctx, builder.GetOptFlag(), logic)
```

在`DoOptimize`函数中，会进行逻辑优化：

```go、、
// planner/core/optimizer.go
logic, err := logicalOptimize(ctx, flag, logic)
if err != nil {
    return nil, err
}
```

```go
// planner/core/optimizer.go
// flag为掩码，代表需要应用哪些优化规则
func logicalOptimize(ctx context.Context, flag uint64, logic LogicalPlan) (LogicalPlan, error) {
    var err error
    for i, rule := range optRuleList {
        if flag&(1<<uint(i)) == 0 {
            continue
        }
        // 遍历优化规则，调用rule.optimize进行优化
        logic, err = rule.optimize(ctx, logic)
        if err != nil {
            return nil, err
        }
    }
    return logic, err
}
```

其中，`optRuleList`为优化规则列表：

```go
// planner/core/optimizer.go
var optRuleList = []logicalOptRule{
    &columnPruner{},
    &buildKeySolver{},
    &aggregationEliminator{},
    &projectionEliminator{},
    &maxMinEliminator{},
    &ppdSolver{},
    &outerJoinEliminator{},
    &aggregationPushDownSolver{},
    &pushDownTopNOptimizer{},
    &joinReOrderSolver{},
}
```

类型`logicalOptRule`为优化规则：

```go
// planner/core/optimizer.go
type logicalOptRule interface {
    optimize(context.Context, LogicalPlan) (LogicalPlan, error)
    name() string
}
```

列表中的每种规则均有相应的`optimize`方法实现，位于`planner/core/rule*`中。

**列裁剪**

列裁剪算法位于`planner/core/rule_column_prunning.go`中。逻辑执行计划`LogicalPlan`接口包含列裁剪`PruneColumns`方法，而每种逻辑执行计划结点均实现了`LogicalPlan`接口。`planner/core/rule_column_prunning.go`则包含了每种逻辑执行计划结点对应的列裁剪`PruneColumns`方法实现。

列裁剪目的为裁剪掉不需要读取的列，以节约IO资源；算法实现为自顶向下遍历逻辑执行计划树，并调用每个结点所实现的`PruneColumns`方法：某个结点需要用到的列，等于它自己需要用到的列，加上父节点需要用到的列：

```go
// lp为逻辑执行计划树的根结点
func (s *columnPruner) optimize(ctx context.Context, lp LogicalPlan) (LogicalPlan, error) {
    err := lp.PruneColumns(lp.Schema().Columns)
    return lp, err
}
```

例如对于`Select`算子，`PruneColumns`方法实现为：

```go
func (p *LogicalSelection) PruneColumns(parentUsedCols []*expression.Column) error {
    child := p.children[0]
    // 父节点用到的列 <= 父节点用到的列 + 当前结点用到的列
    parentUsedCols = expression.ExtractColumnsFromExpressions(parentUsedCols, p.Conditions, nil                )
    // 调用子节点的PruneColumns方法，传入父节点用到的列
    return child.PruneColumns(parentUsedCols)
}
```

**Predicate及Limit下推**

`planner/core/rule_predicate_push_down`中实现了将Predicate下推到Project与Join算子下面。

谓词下推目的为将能下推的条件尽量下推，使得提前过滤更多的记录，减小参与Join等算子的数据量。

谓词下推接口函数为：

```go
func (p *baseLogicalPlan) PredicatePushDown(predicates []expression.Expression) ([]expression.Expression, LogicalPlan)
```

其处理当前的执行计划p，参数predicates表示要添加的过滤条件；函数返回值为无法下推的条件以及新生成的执行计划。

例如，对于Join算子的谓词下推，首先会尽可能将左外连接和右外连接简化为内连接；再收集所有过滤条件，区分哪些是 Join 的等值条件，哪些是 Join 需要用到的条件，哪些全部来自于左子节点，哪些全部来自于右子节点；区分之后，对于内连接，可以把左条件和右条件分别向左右子节点下推。等值条件和其它条件保留在当前的 Join 算子中，剩下的返回。

```go
case InnerJoin:
    tempCond := make([]expression.Expression, 0, len(p.LeftConditions)+len(p.RightConditions)+len(p.EqualConditions)+len(p.OtherConditions)+len(predicates))
    tempCond = append(tempCond, p.LeftConditions...)
    tempCond = append(tempCond, p.RightConditions...)
    tempCond = append(tempCond, expression.ScalarFuncs2Exprs(p.EqualConditions)...)
    tempCond = append(tempCond, p.OtherConditions...)
    tempCond = append(tempCond, predicates...)
    tempCond = expression.ExtractFiltersFromDNFs(p.ctx, tempCond)
    tempCond = expression.PropagateConstant(p.ctx, tempCond)
    dual := Conds2TableDual(p, tempCond)
    if dual != nil {
        return ret, dual
    }
    equalCond, leftPushCond, rightPushCond, otherCond = p.extractOnCondition(tempCond, true, true)
    // 把左条件和右条件分别向左右子节点下推
    p.LeftConditions = nil
    p.RightConditions = nil
    // 等值条件和其它条件保留在当前的 Join 算子中
    p.EqualConditions = equalCond
    p.OtherConditions = otherCond
    leftCond = leftPushCond
    rightCond = rightPushCond
}
leftCond = expression.RemoveDupExprs(p.ctx, leftCond)
rightCond = expression.RemoveDupExprs(p.ctx, rightCond)
leftRet, lCh := p.children[0].PredicatePushDown(leftCond)
rightRet, rCh := p.children[1].PredicatePushDown(rightCond)
addSelection(p, lCh, leftRet, 0)
addSelection(p, rCh, rightRet, 1)
p.updateEQCond()
for _, eqCond := range p.EqualConditions {
    p.LeftJoinKeys = append(p.LeftJoinKeys, eqCond.GetArgs()[0].(*expression.Column))
    p.RightJoinKeys = append(p.RightJoinKeys, eqCond.GetArgs()[1].(*expression.Column))
}
p.mergeSchema()
buildKeyInfo(p)
return ret, p.self
```

**谓词下推算法的执行流程与列裁剪类似：自顶向下遍历执行计划树，在当前结点的`PredicatePushDown`方法中处理谓词下推并调用子节点的`PredicatePushDown`方法。**

在`planner/core/rule_topn_push_down`中，还实现了将Limit下推到Project与Join算子下面。例如：

```go
func (p *LogicalProjection) pushDownTopN(topN *LogicalTopN) LogicalPlan {
    for _, expr := range p.Exprs {
        if expression.HasAssignSetVarFunc(expr) {
            return p.baseLogicalPlan.pushDownTopN(topN)
        }
    }
    if topN != nil {
        for _, by := range topN.ByItems {
            by.Expr = expression.ColumnSubstitute(by.Expr, p.schema, p.Exprs)
        }

        // 删除无意义的常量排序项
        for i := len(topN.ByItems) - 1; i >= 0; i-- {
            switch topN.ByItems[i].Expr.(type) {
            case *expression.Constant:
                topN.ByItems = append(topN.ByItems[:i], topN.ByItems[i+1:]...)
            }
        }
    }
    p.children[0] = p.children[0].pushDownTopN(topN)
    return p
}
```
------


#### 支持的算子

- **UnionScan**

```go
type PhysicalUnionScan struct {
   basePhysicalPlan

   Conditions []expression.Expression

   HandleCol *expression.Column
}
```

> 通过下推 API ，把一部分简单的 SQL 层的执行逻辑下推到 KV 层执行，减少 RPC 的次数和数据传输量，从而提升性能

> 为了解决对脏数据的读取，SQL 层实现了 UnionStore 的结构，UnionStore 对 SQL 层的 Buffer 和 KV 层接口做了一个封装，事务对 KV 层的读写都经过 UnionStore 。
>
> > UnionStore 收到请求时会先在 Buffer 里寻找，找不到时才会调用 KV 层的接口
>
> > 当需要遍历数据的时候 UnionStore 会创建 Buffer 和 KV 的迭代器，并合并成一个

```go
type UnionStore interface {
	MemBuffer
	// GetKeyExistErrInfo gets the key exist error info for the lazy check.
	GetKeyExistErrInfo(k Key) *existErrInfo
    
	// DeleteKeyExistErrInfo deletes the key exist error info for the lazy check.
	DeleteKeyExistErrInfo(k Key)
    
	// WalkBuffer iterates all buffered kv pairs.
	WalkBuffer(f func(k Key, v []byte) error) error
    
	// SetOption sets an option with a value, when val is nil, uses the default value of this option.
	SetOption(opt Option, val interface{})
    
	// DelOption deletes an option.
	DelOption(opt Option)
    
	// GetOption gets an option.
	GetOption(opt Option) interface{}
    
	// GetMemBuffer return the MemBuffer binding to this UnionStore.
	GetMemBuffer() MemBuffer
}
```

> 为了解决 SQL 层脏数据的可见性问题，定义了Union Scan 算法以 Row 为单位，创建一个 DirtyTable 保存事务的修改
>
> > addedRows 保存新写入的 row， deleteRows 保存删除的 row

1. 对于 `INSERT`，我们需要把 row 添加到 addedRows 里。

2. 对于 `DELETE`，我们需要把 row 从 addedRows 里删掉，然后把 row 添加到 deleteRows 里。

3. 对于 `UPDATE`，相当于先执行 `DELETE`, 再执行 `INSERT。`

> 对于每一条下推 API 得到的结果集里的 Row，在 deleteRows 里查找，如果有，那么代表这一条结果已经被删掉，那么把它从结果集里删掉，得到过滤后的结果集。
>
> 把 addedRows 里的所有 Row，放到一个 slice 里，并对这个 slice 用快照结果集相同的顺序排序，生成脏数据结果集。
>
> 返回结果的时候，将过滤后的快照结果集与脏数据结果集进行 Merge。

```go
type DirtyTable struct {bl
	tid int64
	
	addedRows   map[int64]struct{}
	deletedRows map[int64]struct{}
}
```

- **TableScan**

> TableScan 算子实现为 Batch 方式，通过向量化计算加速计算

```go
type PhysicalTableScan struct {
	physicalSchemaProducer

	// AccessCondition is used to calculate range.
	AccessCondition []expression.Expression
	filterCondition []expression.Expression

	Table   *model.TableInfo
	Columns []*model.ColumnInfo
	DBName  model.CIStr
	Ranges  []*ranger.Range
	pkCol   *expression.Column

	TableAsName *model.CIStr

	HandleIdx int

	KeepOrder bool
	Desc      bool
}
```



- **IndexScan**

> 全表扫描(索引数据)

```go
type PhysicalIndexScan struct {
   physicalSchemaProducer

   // AccessCondition is used to calculate range.
   AccessCondition []expression.Expression

   Table      *model.TableInfo
   Index      *model.IndexInfo
   IdxCols    []*expression.Column
   IdxColLens []int
   Ranges     []*ranger.Range
   Columns    []*model.ColumnInfo
   DBName     model.CIStr

   TableAsName *model.CIStr

   // dataSourceSchema is the original schema of DataSource. The schema of index scan in KV and index reader in TiDB
   // will be different. The schema of index scan will decode all columns of index but the TiDB only need some of them.
   dataSourceSchema *expression.Schema

   Desc      bool
   KeepOrder bool
   // DoubleRead means if the index executor will read kv two times.
   // If the query requires the columns that don't belong to index, DoubleRead will be true.
   DoubleRead bool
}
```



- **Projection**

```go
type ProjectionExec struct {
	baseExecutor

	evaluatorSuit *expression.EvaluatorSuite

	prepared    bool
	finishCh    chan struct{}
	outputCh    chan *projectionOutput
	fetcher     projectionInputFetcher
	numWorkers  int64
	workers     []*projectionWorker
	childResult *chunk.Chunk

	wg sync.WaitGroup

	parentReqRows int64
}
```



- **Filter**

```go
// Filter the input expressions, append the results to result.
func Filter(result []Expression, input []Expression, filter func(Expression) bool) []Expression {
	for _, e := range input {
		if filter(e) {
			result = append(result, e)
		}
	}
	return result
}
```



- **Aggregate**

> 这里的聚合算法为 Hash Aggregate 
>
> > 在Hash Aggregate 的计算过程中，Hash 表的键为聚合计算的`Group-By`列，值为聚合函数的中间结果`sum`和`count`。
> >
> > 输入数据输入完毕之后，扫描 Hash 表并计算，便可得到结果

> 由于对分布式计算的需要，聚合函数有以下计算模式

| AggFunctionMode |  输入值  | 输出值                 |
| --------------- | :------: | ---------------------- |
| CompleteMode    | 原始数据 | 最终结果               |
| FinalMode       | 中间结果 | 最终结果               |
| Partial1Mode    | 原始数据 | 中间结果               |
| Partial2Mode    | 中间结果 | 进一步聚合后的中间结果 |



- **HashJoin**

> 具有以下任务
>
> > Main Thread
> >
> > > 读取所有的 Inner 表数据
> >
> > > 根据 Inner 表数据构造哈希表
> >
> > > 启动 Outer Fetcher 和 Join Worker 开始后台工作，生成 Join 结果，各个 goroutine 的启动过程由 `fetchAndProbeHashTable` 这个函数完成
> >
> > > 将 Join Worker 计算出的 Join 结果返回给 `NextChunk` 接口的调用方法
>
> > Outer Thread: 负责读取 Outer 表的数据并分发给各个 Join Worker
>
> > Join Work: 负责查哈希表、Join 匹配的 Inner 和 Outer 表的数据，并把结果传递给 Main Thread

```go
type HashJoinExec struct {
   baseExecutor

   outerSideExec     Executor
   innerSideExec     Executor
   innerSideEstCount float64
   outerSideFilter   expression.CNFExprs
   outerKeys         []*expression.Column
   innerKeys         []*expression.Column

   // concurrency is the number of partition, build and join workers.
   concurrency  uint
   rowContainer *hashRowContainer
   // joinWorkerWaitGroup is for sync multiple join workers.
   joinWorkerWaitGroup sync.WaitGroup
   // closeCh add a lock for closing executor.
   closeCh  chan struct{}
   joinType plannercore.JoinType

   // We build individual joiner for each join worker when use chunk-based
   // execution, to avoid the concurrency of joiner.chk and joiner.selected.
   joiners []joiner

   outerChkResourceCh chan *outerChkResource
   outerResultChs     []chan *chunk.Chunk
   joinChkResourceCh  []chan *chunk.Chunk
   joinResultCh       chan *hashjoinWorkerResult

   prepared bool
}
```

支持三种 Join 方式

1. leftOuterJoiner
2. rightOuterJoiner
3. innerJoiner



- **MergeJoin**

```go
type MergeJoinExec struct {
   baseExecutor

   stmtCtx      *stmtctx.StatementContext
   compareFuncs []expression.CompareFunc
   joiner       joiner
   isOuterJoin  bool

   prepared bool
   outerIdx int

   innerTable *mergeJoinInnerTable
   outerTable *mergeJoinOuterTable

   innerRows     []chunk.Row
   innerIter4Row chunk.Iterator

   childrenResults []*chunk.Chunk
}
```



- **Limit**

```go
type LimitExec struct {
   baseExecutor

   begin  uint64
   end    uint64
   cursor uint64

   // meetFirstBatch represents whether we have met the first valid Chunk from child.
   meetFirstBatch bool

   childResult *chunk.Chunk
}
```



- **Selection**

```go
type selectionExec struct {
   conditions        []expression.Expression
   relatedColOffsets []int
   row               []types.Datum
   evalCtx           *evalContext
   src               executor
}
```



- **Sort**

```go
type SortExec struct {
   baseExecutor

   ByItems []*plannercore.ByItems
   Idx     int
   fetched bool
   schema  *expression.Schema

   // keyColumns is the column index of the by items.
   keyColumns []int
   // keyCmpFuncs is used to compare each ByItem.
   keyCmpFuncs []chunk.CompareFunc
   // rowChunks is the chunks to store row values.
   rowChunks *chunk.List
   // rowPointer store the chunk index and row index for each row.
   rowPtrs []chunk.RowPtr
}
```



- **TopN**

> 将相邻的 Limit 算子和 Sort 算子组合成 TopN 算子节点，表示某个排序规则提取记录的前 N 项

```go
type TopNExec struct {
   SortExec
   limit      *plannercore.PhysicalLimit
   totalLimit uint64

   chkHeap *topNChunkHeap
}
```



- **TableReader**

> 将 TiKV 上底层扫表算子 `TableFullScan `或 `TableRangeScan ` 得到的数据进行汇总

```go
type TableReaderExecutor struct {
   baseExecutor

   table  table.Table
   ranges []*ranger.Range
   // kvRanges are only use for union scan.
   kvRanges []kv.KeyRange
   dagPB    *tipb.DAGRequest
   startTS  uint64
   // columns are only required by union scan and virtual column.
   columns []*model.ColumnInfo

   // resultHandler handles the order of the result. Since (MAXInt64, MAXUint64] stores before [0, MaxInt64] physically
   // for unsigned int.
   resultHandler *tableResultHandler
   plans         []plannercore.PhysicalPlan

   keepOrder bool
   desc      bool
}
```



- **IndexReader**

> 将 TiKV 上底层扫表算子 `IndexFullScan` 或 `IndexRangeScan `得到的数据进行汇总

```go
type IndexReaderExecutor struct {
   baseExecutor

   // For a partitioned table, the IndexReaderExecutor works on a partition, so
   // the type of this table field is actually `table.PhysicalTable`.
   table           table.Table
   index           *model.IndexInfo
   physicalTableID int64
   keepOrder       bool
   desc            bool
   ranges          []*ranger.Range
   // kvRanges are only used for union scan.
   kvRanges []kv.KeyRange
   dagPB    *tipb.DAGRequest
   startTS  uint64

   // result returns one or more distsql.PartialResult and each PartialResult is returned by one region.
   result distsql.SelectResult
   // columns are only required by union scan.
   columns []*model.ColumnInfo
   // outputColumns are only required by union scan.
   outputColumns []*expression.Column

   idxCols []*expression.Column
   colLens []int
   plans   []plannercore.PhysicalPlan
}
```



- **IndexLookUpReader**

> 汇总 Build 端 TiKV 扫描上来的 RowID，再去 Probe 端上根据这些 `RowID` 精确地读取 TiKV 上的数据。Build 端是 `IndexFullScan` 或 `IndexRangeScan` 类型的算子，Probe 端是 `TableRowIDScan` 类型的算子。

```go
type IndexLookUpExecutor struct {
   baseExecutor

   table     table.Table
   index     *model.IndexInfo
   keepOrder bool
   desc      bool
   ranges    []*ranger.Range
   dagPB     *tipb.DAGRequest
   startTS   uint64
   // handleIdx is the index of handle, which is only used for case of keeping order.
   handleIdx    int
   tableRequest *tipb.DAGRequest
   // columns are only required by union scan.
   columns []*model.ColumnInfo
   *dataReaderBuilder
   // All fields above are immutable.

   idxWorkerWg sync.WaitGroup
   tblWorkerWg sync.WaitGroup
   finished    chan struct{}

   kvRanges      []kv.KeyRange
   workerStarted bool

   resultCh   chan *lookupTableTask
   resultCurr *lookupTableTask

   idxPlans []plannercore.PhysicalPlan
   tblPlans []plannercore.PhysicalPlan
   idxCols  []*expression.Column
   colLens  []int
}
```
------
#### TinyKV
##### 大致实现

TinyKV使用**有序的数组的排列**，可以看作是一个提供了如下性质的 KV 引擎：

- Key 和 Value 都是 bytes 数组，也就是说无论原先的类型是什么，我们都要序列化后再存入
- Scan(startKey)，任意给定一个 Key，这个接口可以按顺序返回所有大于等于这个 startKey 数据。
- Set(key, value)，将 key 的值设置为 value。

##### 具体实现

- ##### 逻辑结构

对每个表分配一个 TableID，每一行分配一个 RowID（如果表有整数型的 Primary Key，那么会用 Primary Key 的值当做 RowID），其中 TableID 在整个集群内唯一，RowID 在表内唯一，这些 ID 都是 int64 类型。 每行数据按照如下规则进行编码成 Key-Value pair：

```
    Key： tablePrefix_tableID_recordPrefixSep_rowID
    Value: [col1, col2, col3, col4]
```

- ##### 存储结构

TinyKV使用LSM树进行读写，相比 B+ 树以牺牲读性能的代价在写入性能上获得了较大的提升，同时其特性也使得 TinyKV 可以有序的存储 KV pair，为上层 TinySql 的存储提供很大方便。

LSM树主要有三个组成部分：
![image](https://github.com/bytedance-training-camp/Youth-training-camp-defense-document/blob/main/img/QQ%E5%9B%BE%E7%89%8720220823095201.png)
***1) MemTable***

MemTable是在**内存**中的数据结构，用于保存最近更新的数据，会按照Key有序地组织这些数据，LSM树对于具体如何组织有序地组织数据并没有明确的数据结构定义，例如Hbase使跳跃表来保证内存中key的有序。

因为数据暂时保存在内存中，内存并不是可靠存储，如果断电会丢失数据，因此通常会通过WAL(Write-ahead logging，预写式日志)的方式来保证数据的可靠性。

***2) Immutable MemTable***

当 MemTable达到一定大小后，会转化成Immutable MemTable。Immutable MemTable是将转MemTable变为SSTable的一种中间状态。写操作由新的MemTable处理，在转存过程中不阻塞数据更新操作。

***3) SSTable(Sorted String Table)***

**有序键值对**集合，是LSM树组在**磁盘**中的数据结构。为了加快SSTable的读取，可以通过建立key的索引以及布隆过滤器来加快key的查找。

##### 项目实现

服务启动时调用registerStores()函数注册存储引擎tikv和mocktikv，调用createStoreAndDomain()函数根据提供的参数（如未提供则使用默认值）创建存储引擎。

在 TinySql 中，事务的执行过程会被缓存在 buffer 中，在提交时，才会通过 Percolator 提交协议将其完整的写入到分布的 TiKV 存储引擎中。这一调用的入口是 `store/tikv/txn.go` 中的 `tikvTxn.Commit` 函数。

```go
func (txn *tikvTxn) Commit(ctx context.Context) error {
   if !txn.valid {
      return kv.ErrInvalidTxn
   }
   defer txn.close()

   failpoint.Inject("mockCommitError", func(val failpoint.Value) {
      if val.(bool) && kv.IsMockCommitErrorEnable() {
         kv.MockCommitErrorDisable()
         failpoint.Return(errors.New("mock commit error"))
      }
   })

   // connID is used for log.
   var connID uint64
   val := ctx.Value(sessionctx.ConnID)
   if val != nil {
      connID = val.(uint64)
   }

   var err error
   committer := txn.committer
   if committer == nil {
      committer, err = newTwoPhaseCommitter(txn, connID)
      if err != nil {
         return errors.Trace(err)
      }
   }
   if err := committer.initKeysAndMutations(); err != nil {
      return errors.Trace(err)
   }
   if len(committer.keys) == 0 {
      return nil
   }

   err = committer.execute(ctx)
   return errors.Trace(err)
}
```

在执行时，一个事务可能会遇到其他执行过程中的事务，此时需要通过 Lock Resolve 组件来查询所遇到的事务状态，并根据查询到的结果执行相应的措施。

##### percolator 协议

- percolator 分为 prewrite 和 commit 阶段
- 为了防止出现数据部分可见等情况，在 prewrite 阶段实际进行写数据，在 commit 阶段才让数据对外可见
- rollback keys 用于回滚事务
- check TxnStatus 通过查询 Lock 所属的 Primary Key 来判断事务的状态
- resolve locks 根据 check TxnStatus 得到的结果进行相应处理

因为 tinykv 底层是 multi-raft 的实现，所以是分多 region 的，而且 key 是由 range 来分区的，因此首先需要将 keys 进行划分。该划分在 GroupKeysByRegion 函数中实现，只需要调用 LocateKey 并进行一些额外判断

percolator 的第一阶段是 prewrite， 这里 tinysql 的 prewrite 会向 tinykv 层的 percolator 发送一个 prewrite 请求，进行数据的写入，在 Prewrite 阶段，对于一个 Key 的操作会写入两条记录:

- Default CF 中存储了实际的 KV 数据。
- Lock CF 中存储了锁，包括 Key 和时间戳信息，会在 Commit 成功时清理。

###### prewrite phase

- 首先 buildPrewriteRequest

- 然后发送请求到 tinykv 层

- 判断 regionErr

- 如果出现 keyErrs ，则解析出对应的冲突 locks，然后进行 ResolveLocks

- - 在prewrite的时候，如果有一些由其他事务留下的重叠锁，tinykv 将返回 keyErr。这些事务的状态是不明确的。ResolveLocks 将通过锁来检查事务的状态并解决它们。

###### commit phase

如果prewrite 成功了，需要进入第二阶段：commit phase。该阶段 tinysql 会向 tinykv 发送 commit request，先将主键所在的group 进行提交，如果主键提交成功，我们就标记这次事务提交成功，然后再进行其他 group 的并行提交（该过程在2pc.go的 twoPhaseCommitter.doActionOnKeys中实现）：

- 首先 buildCommitRequest
- 然后发送请求到 tinykv 层
- 对各自可能的错误返回进行特定处理
- 设置 committed 为 true

###### rollback

如果 prewrite 失败了，或者由于其他原因事务执行失败，就需要进行 rollback：

- 首先 buildCleanupRequest
- 然后发送请求到 tinykv 层
- 对各自可能的错误返回进行特定处理

##### 总结

check txnstatus 是 resolve lock的基础，check txn status可以去确认一个键是否有锁、锁的状态以及该键的主键状态（提交or回滚）。在 Percolator 协议下，会通过查询 Lock 所属的 Primary Key 来判断事务的状态：

- 首先还是先建立 request
- 然后判断主键的 region 并发送请求
- 对各自可能的错误返回进行特定处理
- 如果 cmdResp.LockTtl != 0 就设置 status 的 ttl，否则就设置 status 的 commitTS 为 cmdresp.commitTS

我们可以通过checkTxnStatus来确定事务的状态，如果主键提交了，可以得到其 commitTs，如果主键还在提交中，可以得到lock信息来判断锁是否过期 ，如果是 rollback，可以得到 commitTs为0。因此我们就可以通过事务的这些status来判断如何处理锁：

- Lock Resolver 的职责就是当一个事务在提交过程中遇到 Lock 时，需要如何应对:
  当一个事务遇到 Lock 时，可能有几种情况。

- - Lock 所属的事务还未提交这个 Key，Lock 尚未被清理；
  - Lock 所属的事务遇到了不可恢复的错误，正在回滚中，尚未清理 Key；
  - Lock 所属事务的节点发生了意外错误，例如节点 crash，这个 Lock 所属的节点已经不能够更新它。

在 Percolator 协议下，会通过查询 Lock 所属的 Primary Key 来判断事务的状态， 但是当读取到一个未完成的事务（Primary Key 的 Lock 尚未被清理）时，我们所期望的， 是等待提交中的事物至完成状态，并且清理如 crash 等异常留下的垃圾数据。 此时会借助 ttl 来判断事务是否过期，遇到过期事务时则会主动 Rollback 它。

对于 get 函数，需要补充的只是：从 keyErr 中解析出锁，然后进行 resolveLock

------































## 四、测试结果



























## 五、演示Demo



































## 六、项目总结与反思



