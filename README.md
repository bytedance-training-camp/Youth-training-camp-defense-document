# **青训营结业项目答辩汇报**

## 一、项目介绍
- 项目概括：实现一个简易分布式计算系统最核心的部分：MySQL协议、SQL解析、验证、优化器、执行器、多种支持算子等等，在这里我们不考虑垃圾回收、高并发安全等拓展功能。

- 项目服务地址:
- Github地址：https://github.com/bytedance-training-camp/simple-distributed-sql-engine
## 二、项目分工











## 三、项目实现

### 3.1 技术选型与相关开发文档

















### 3.2架构设计



















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
![image](https://github.com/bytedance-training-camp/-/blob/main/img/img/%E5%9B%BE%E7%89%871.png)

------


#### SQL解析与验证
![image](https://github.com/bytedance-training-camp/-/blob/main/img/img/QQ%E5%9B%BE%E7%89%8720220821094807.jpg)

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
![image](https://github.com/bytedance-training-camp/-/blob/main/img/img/image-20220818150800016.png)

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

![执行器树](https://img1.www.pingcap.com/prod/1_c3e07627e9.png)

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






























## 四、测试结果



























## 五、演示Demo



































## 六、项目总结与反思



