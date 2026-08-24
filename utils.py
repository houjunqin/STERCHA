# log string
def log_string(log_dir, string):
    with open(log_dir + 'log', 'a+') as log:
        log.write(string + '\n')
        log.flush()
        print(string)