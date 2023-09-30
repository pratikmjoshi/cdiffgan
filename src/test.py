import sys                                                                  

number = int(sys.argv[1])                                                        

if number <= 1:                                                            
    print(number)                                                                    
else:
    print('error: {}'.format(number))
    raise ValueError('Number is higher than 10!')                           
    sys.exit(1)  
