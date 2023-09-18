#include<stdio.h>
#include<math.h>
struct floor
{
    int delay;              //延迟到达时间
    int goal;               //目标楼层
    int InElv;              //判断是否在电梯内
    int pas;                //判断楼层是否还有人
};
int main()
{
    struct floor m[11]={0};            //罗列每层楼的情况
    int elv;
    scanf("elevator %d",&elv);
    int a,b,c,n=0;
    printf("请输入每一位呼叫用户的起始楼层，目标楼层以及其到达时间：\nPs:输入英文符号结束输入\n");
    int up=0,down=0;                //由于电梯起始楼层人为设定，可以考虑优先向人多的地方运行以提高效率
    while(scanf("%d %d %d\n",&a,&b,&c))
    {
        m[a].goal=b;
        m[a].pas=1;
        m[a].delay=c;
        n++;
        if(a>elv) ++up;
        if(a<elv) ++down;             //记录电梯上方和下方呼唤的人数
    }
    printf("The detailed progcess is as followed:\n");
    int way=1,key=0;                       //way声明电梯的运行方式，即判断是网上还是往下   key声明是否抵达关键楼层   
    int time=0,num=0;                           //记录运行时间与电梯内的人数 
    if (down>up) way=-1;              //优先选择人多的方向            
    while(n>0)                       //对电梯的运行进行模拟
    {
        if(time==0) key=1;                    //启动时刻是一个关键楼层
        if(m[elv].pas==1&&num<4&&time>=m[elv].delay)          //每到一层楼判断该层是否有人，能否进入电梯
        {
            ++num;
            m[elv].pas=0;                //乘客已经接走
            m[elv].InElv=1;
            key=1;
        }
        for(int i=1;i<=10;++i)
        {
            if(m[i].goal==elv&&m[i].InElv==1)            //判断电梯内是否有人到达目的地
            {
                --num;
                m[i].InElv=0;               //乘客已经送达
                key=1;
                --n;                           //还剩n人需要抵达目的地
            }
        }
        if(key==1)
        {
            printf("%d %d %d\n",elv,time,num);
        }
        int target1=0,target2=0;                                  //上方和下方的目标
        for(int i=1;i<=10;i++)                                  //将SCAN策略优化为LOOK，减少无用的循环时间
        {
            if((m[i].goal<elv&&m[i].InElv==1)||(i<elv&&m[i].pas==1&&num<4)) ++target1;   //记录电梯下方目标 Ps：下方有目标的条件为下方有需要送达的人或者下方有可以接取的人。
            if((m[i].goal>elv&&m[i].InElv==1)||(i>elv&&m[i].pas==1&&num<4)) ++target2;   //记录电梯上方目标 上方同理。
        }
        if (target2==0&&way==1) way=-1;               //如果电梯向上运行且上方无目标，则改为向下运行
        if (target1==0&&way==-1) way=1;               //如果电梯向下运行且下方无目标，则改为向上运行
        elv+=way;
        ++time;
        key=0;
    }
    return 0;
}