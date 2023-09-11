#include<stdio.h>
#include<math.h>
int main()
{
    freopen("chart.txt","r",stdin);
    freopen("result.txt","w",stdout);
    int elv,now,goal;
    scanf("elevator %d\n%d %d",&elv,&now,&goal);
    int time=0;
    printf("%d 0 0\n",elv);
    time+=abs(now-elv);     //time的计算：模拟运行过程的计时环节
    printf("%d %d 1\n",now,time);
    time+=abs(now-goal);
    printf("%d %d 0\n",goal,time);
    fclose(stdin);fclose(stdout);
    return 0;
}