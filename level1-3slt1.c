#include<stdio.h>
#include<math.h>
struct floor                              //对每层楼进行标记
    {
        bool pas;                        //判断该层是否有人
        int goal;                        //声明如果有人的话他的目标楼层
        bool InElv ;                     //判断该楼是否有人进入电梯
    };                                           
floor m[11]; 
int main()
{
    int num=0,time=0;
    int elv=1;
    printf("Please input the present floor and the goal of each passenger:\n");
    int a,b,n=0,top=0,down=0;
    while(scanf("%d %d",&a,&b)==1)
    {
        floor[a].goal=b;
        floor[a].pas=1;                      //声明该楼层有人
        ++n;                          //统计总共有多少人呼叫。ps：虽然提前记录了目标楼层，但是不影响电梯的模拟
    }
    int way=1,key=0;                       //way声明电梯的运行方式，即判断是网上还是往下   key声明是否抵达关键楼层   
    int time=0;                           //记录运行时间             
    while(n>0)                       //对电梯的运行进行模拟
    {
        if(time==0) key=1;                    //启动时刻是一个关键楼层
        if(floor[elv].pas==1&&num<4)          //每到一层楼判断该层是否有人，能否进入电梯
        {
            ++num;
            floor[elv].pas=0;                //乘客已经接走
            floor[elv].InElv=1;
            key=1;
        }
        for(int i=1;i<=10;++i)
        {
            if(floor[i].goal==elv&&floor[i].InElv==1)            //判断电梯内是否有人到达目的地
            {
                --num;
                floor[i].InElv=0;               //乘客已经送达
                key=1;
                --n;                           //还剩n人需要抵达目的地
            }
        }
        if(key==1)
        {
            printf("%d %d %d\n",elv,time,num);
        }
        if(elv==10) way=-1;
        if(elv==1)  way=1;               //判断电梯运行策略
        elv+=way;
        ++time;
    }
    return 0;
}