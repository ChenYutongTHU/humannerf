import time
import sys, os
print(sys.argv[1], sys.argv[2], 'Sleep 5 seconds')
time.sleep(5)
print(sys.argv[1], sys.argv[2], 'End')
os.system(f'touch debug_output/{sys.argv[1]}.txt')