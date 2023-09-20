//出于考虑模拟真实电梯的呼叫方式的一种策略的改进
#include<stdio.h>
struct floor
{
    int delay;              //延迟到达时间
    int goal;               //目标楼层
    int InElv;              //判断是否在电梯内,以及乘客在哪个电梯里面
    int pas;                 //用于标记是否有乘客以及乘客的乘坐方向

};
int wait=0,time=0;                                 //声明已经呼叫并在等待的人数以及电梯运行时间
void look(struct floor *m,int *way,int num,int *elv)             //用于检索两个电梯此时的运行目标确定运行方向，动态调整
{
    int target1=0,target2=0;                                  //上方和下方的目标
    for(int i=1;i<=10;i++)                            
    {
        if((m[i].goal<*elv&&m[i].InElv==1)||(i<=*elv&&m[i].pas!=0&&num<4&&((time+1)>=m[i].delay))) ++target1;   //记录电梯下方目标 Ps：下方有目标的条件为下方有需要送达的人或者下方有可以接取的人。
        if((m[i].goal>*elv&&m[i].InElv==1)||(i>=*elv&&m[i].pas!=0&&num<4&&((time+1)>=m[i].delay))) ++target2;   //记录电梯上方目标 上方同理。
    }
    if (target2==0&&*way==1) *way=-1;               //如果电梯向上运行且上方无目标，则改为向下运行
    if (target1==0&&*way==-1) *way=1;               //如果电梯向下运行且下方无目标，则改为向上运行
    if(wait>0||num>0) *elv+=*way;
}
void check(int *way,int elv,struct floor *m,int *key,int *n,int *num,int mark)                   //用于两个电梯到达楼层后的检索功能
{

    if(*num<4&&(*way==m[elv].pas)&&(time+1)>=m[elv].delay)
    {
        ++*num;
        m[elv].pas=0;                //乘客已经接走
        --wait; //接走一名乘客，此时等待名单减少一人
        m[elv].InElv=mark;   
        *key=1;
    }
    for(int i=1;i<=10;++i)
    {
        if(m[i].goal==elv&&m[i].InElv==mark)            //判断电梯内是否有人到达目的地
        {
            --*num;
            m[i].InElv=0;               //乘客已经送达
            *key=1;
            --*n;                           //还剩n人需要抵达目的地
        }
    }
}
int main()
{
    struct floor m[11]={0};
    int elv1,elv2;
    printf("请输入两个电梯各种的起始楼层\n一号电梯:\n");
    scanf("%d",&elv1);
    printf("二号电梯:\n");
    scanf("%d",&elv2);
    int n=0,a,b,c;
    while(scanf("%d %d %d\n",&a,&b,&c))
    {
        m[a].goal=b;
        m[a].delay=c+1;                               //为了区分电梯开始运行时就已经呼叫的乘客与空楼层，选择让每个乘客的延迟时间+1作为标记
        ++n;
        if(b>a) m[a].pas=1;
        if(b<a) m[a].pas=-1;            //判断乘客的运行方向
    }
    int way1,way2,mark1=1,mark2=2;       //两个电梯的运行方式，且可用于标记电梯接取向上还是向下的乘客 PS:mark用于标记是一号还是二号电梯
    if (elv1>elv2)  way1=-1;
    else way1=1;
    way2=-way1;            //保证两个电梯初始运动时能够扫到全部楼层
    int key1=1,key2=1,num1=0,num2=0;       //电梯内的人数以及判断是否为关键楼层
    while(n>0)
    {
        for(int i=1;i<=10;++i)                   //依次寻找此时开始呼叫的乘客
        {
            if((time+1)==m[i].delay) ++wait;        //若此时乘客到达并呼叫，将加入等待名单中
        }
        check(&way1,elv1,m,&key1,&n,&num1,mark1);
        check(&way2,elv2,m,&key2,&n,&num2,mark2);
        if (key1==1) printf("elevator1:%d %d %d\n",elv1,time,num1);
        if (key2==1) printf("elevator2:%d %d %d\n",elv2,time,num2);
        if (wait>0||(num1+num2)>0)                              //如果此时开始有人进行呼叫或者电梯内还有人，电梯开始运行（两辆电梯同时响应）
        {
            look(m,&way1,num1,&elv1);
            look(m,&way2,num2,&elv2);               
        }
        ++time;
        key1=key2=0;
    }
    return 0;
}