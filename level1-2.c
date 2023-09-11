#include<stdio.h>
#include<math.h>
int main()
{
    FILE *fin=NULL,*fout=NULL;
    fin=fopen("chart.txt","rb");
    fout=fopen("result.txt","wb");
    int now,elv,goal;
    fscanf(fin,"elevator %d\n%d %d",&elv,&now,&goal);
    if(fin==NULL)printf("ERROR!");    //判断文件是否读取成功
    else
    {
        int time=0;
        fprintf(fout,"%d 0 0\n",elv);
        time+=abs(now-elv);     //time的计算：模拟运行过程的计时环节
        fprintf(fout,"%d %d 1\n",now,time);
        time+=abs(now-goal);
        fprintf(fout,"%d %d 0\n",goal,time);
    }
    fclose(fin);fclose(fout);
    return 0;
}