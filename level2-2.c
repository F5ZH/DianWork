#include<stdio.h>
#include<stdlib.h>
struct floor
{
    int delay;              //延迟到达时间
    int goal;               //目标楼层
    int InElv;              //判断是否在电梯内,以及乘客在哪个电梯里面
    int pas;                 //用于标记是否有乘客以及乘客的乘坐方向

};
int time=0;
void check(int *way,int elv,struct floor *m,int *key,int *n,int *num,int mark)                   //用于两个电梯到达楼层后的检索功能
{
    if(*num<4&&(*way==m[elv].pas)&&time>m[elv].delay)
    {
        ++*num;
        m[elv].pas=0;                //乘客已经接走
        if (mark==1) m[elv].InElv=1;
        if (mark==2) m[elv].InElv=2;    
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
    if(elv==10) *way=-1;
    if(elv==1) *way=1;
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
        m[a].delay=c;
        ++n;
        if(b>a) m[a].pas=1;
        if(b<a) m[a].pas=-1;
    }
    int way1,way2,mark1=1,mark2=2;       //两个电梯的运行方式，且可用于标记电梯接取向上还是向下的乘客 PS:mark用于标记是一号还是二号电梯
    if (elv1>elv2)  way1=-1;
    else way1=1;
    way2=-way1;            //保证两个电梯初始运动时能够扫到全部楼层
    int key1=1,key2=1,num1=0,num2=0;       //运行的时间,电梯内的人数以及判断是否为关键楼层
    while(n>0)
    {
        check(&way1,elv1,m,&key1,&n,&num1,mark1);
        check(&way2,elv2,m,&key2,&n,&num2,mark2);
        if (key1==1) printf("elevator1:%d %d %d\n",elv1,time,num1);
        if (key2==1) printf("elevator2:%d %d %d\n",elv2,time,num2);
        elv1+=way1;
        elv2+=way2;
        ++time;
        key1=key2=0;
    }
    return 0;
}