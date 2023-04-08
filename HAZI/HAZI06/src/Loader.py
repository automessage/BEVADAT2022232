import time
from threading import Thread

class Loader:
    def __init__(self, msg:str = 'Töltés'):
        self._running = True
        self._msg = msg
        self._time = 0.0

    def terminate(self):
        self._running = False
        print("\r", end="")
        print('Befejezve ' + str(self._time) + ' mp alatt')
    
    def start(self):
        Thread(target=self.__run, daemon=True).start()

    def __run(self):
        i = 0
        direction = 1
        loadStates = ['.   ', '..  ', '...  ', '....', ' ...', '  ..', '   .']
        print(self._msg)
        while self._running:
            self._time += 0.5
            print("\r", end="")
            print(loadStates[i], end="")
            
            if i + direction < 0 or i + direction == len(loadStates):
                direction = direction * -1
            
            i = i + direction
            time.sleep(0.5)


# t = Thread(target = loading.run, daemon=True)
# t.start()

# time.sleep(10)

# loading.terminate()