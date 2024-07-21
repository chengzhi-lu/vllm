declare -a array

# 初始化数组
array[0]="element1_sub1 element1_sub2"
array[1]="element2_sub1 element2_sub2"
array[2]="element3_sub1 element3_sub2"

# 访问数组元素
for i in "${!array[@]}"; do
    element=(${array[i]})  # 将字符串拆分为数组
    sub1=${element[0]}
    sub2=${element[1]}
    echo "Element $i: sub1=$sub1, sub2=$sub2"
done

for i in "${array[@]}"; do
    
    element=(${i})  # 将字符串拆分为数组
    echo "Element: sub1=${element[0]}, sub2=${element[1]}"

done