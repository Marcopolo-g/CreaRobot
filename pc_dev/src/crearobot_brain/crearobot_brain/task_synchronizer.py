import time
import asyncio
import concurrent.futures

class TaskSynchronizer():
    """
    Synchroniseur corrigé : ne bloque pas le flux ROS pour éviter l'auto-écho.
    """

    def __init__(self, max_workers=5):
        self.loop = asyncio.get_event_loop()
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers)

    def __worker(self, *args):
        delay_exe = args[0][0]
        func = args[0][1]
        if delay_exe > 0:
            time.sleep(delay_exe)
        return func()

    async def __non_blocking(self, tasks):
        fs = []
        # On note l'heure précise du début d'envoi
        t_start = time.time()
        
        for task in tasks:
            delay = task[0]
            func = task[1]

            # LOGIQUE DE SOUSTRACTION :
            # Si c'est une tâche différée (le Stop Bouche), on retire le temps
            # que le code a déjà mis à traiter les tâches précédentes.
            if delay > 0:
                elapsed = time.time() - t_start
                delay = max(0, delay - elapsed)
            
            # On envoie à l'executor SANS ATTENDRE (comme ta version originale)
            fs.append(self.loop.run_in_executor(
                self.executor, self.__worker, (delay, func)))
                
        done, pending = await asyncio.wait(fs=fs, return_when=asyncio.ALL_COMPLETED)
        results = [task.result() for task in done]
        return results

    def sync(self, tasks):
        # On reste sur la structure loop.run_until_complete
        results = self.loop.run_until_complete(self.__non_blocking(tasks))
        return results