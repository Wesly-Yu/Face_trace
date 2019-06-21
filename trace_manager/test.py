# a='abcdefg'
# # b=''.join(i for i in a[::-1])
# # # 	b=i+b
# # # b=a[::-1]
# b= list(a)
# b.reverse()
# b=''.join(b)
# print(b)



def quick_sort(sort_list):
    if sort_list ==[]:
        return []
    else:
        first = sort_list[0]
        #推导式实现
        less = quick_sort([l for l in sort_list[1:] if l <first])
        more = quick_sort([m for m in sort_list[1:] if m >=first])
        return less+[first]+more
list = quick_sort([1,32,100,7,21,6,66,33,11,55,77,61])
print(list)