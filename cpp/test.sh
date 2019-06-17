#!/bin/bash

# Define a timestamp function
timestamp() {
  date +"%T"
}

# do something...

# declare -i h1
# declare -i m1
# declare -i s1

h1=`date +%H` # print timestamp
m1=`date +%M`
s1=`date +%S`
sum1=$((($h1*60*60)+($m1*60)+($s1)))
timestamp # print another timestamp
echo $sum1

./a.out > images/cover_hq_ns100.ppm

h2=`date +%H` # print timestamp
m2=`date +%M`
s2=`date +%S`
sum2=$((($h2*60*60)+($m2*60)+($s2)))
timestamp # print another timestamp
echo $sum2
diff=$(($sum2-$sum1))
echo $diff

# declare -i num
# num=03
# num2=$(($num + 5))
# echo $num $num2
