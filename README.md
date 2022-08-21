# **青训营结业项目答辩汇报**

## 一、项目介绍









## 二、项目分工











## 三、项目实现

#### 3.1 技术选型与相关开发文档

















#### 3.2架构设计



















#### 3.3项目代码介绍

- 用户界面-网络协议

关于如何实现MySQL协议的代码都放在server包中。启动server时，首先在协议层入口，有一个Go retinue监听端口，等待从客户端发来的包，并对发来的包做处理，在server文件夹的conn中，这里可以认为是分布式存储系统的入口。首先在clientConn Run( )中，这里会在一个循环中，不断读取网络包。

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






























## 四、测试结果



























## 五、演示Demo



































## 六、项目总结与反思



