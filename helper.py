import functools
import time

def retry(func):
        """
        
        retries executing the same function if it falils for 5 times after waiting for a second. 

        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            count = 0
            while count < 5:
                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    print(f"{func.__name__} failed with error: {str(e)}")
                    count += 1
                    time.sleep(2)
            raise Exception(f"{func.__name__} failed after 5 attempts.")
        
        return wrapper

def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print('Elapsed time: {} seconds'.format(end_time - start_time))
        return result
    return wrapper