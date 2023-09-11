#include<stdio.h>
#include<math.h>
int main(){
    int now,goal,elv;      //分别设置乘客所在层数，目标以及电梯所在层数
    scanf("%d %d %d",&now,&goal,&elv);
    int time=0;
    printf("%d 0 0\n",elv);
    time+=abs(now-elv);     //time的计算：模拟运行过程的计时环节，其实也可以直接在输出时计算（）
    printf("%d %d 1\n",now,time);
    time+=abs(now-goal);
    printf("%d %d 0\n",goal,time);
    return 0;
}
