n_list = { 1, 2, 3, 4, 5, 6, 7, 8, 9}
for a in n_list:
    for b in n_list:
        msg = f'{a} x {b} = {a*b}'
        print(msg)
        print('-' * 10)
    print('='*10)
print('~'*10)

