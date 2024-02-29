import pickle
import os

class Persist:
    @classmethod
    def overwrite(cls, fname) -> bool:
        ask = lambda: input(f'{fname} exists. Overwrite? [y/n]')
        if os.path.exists(fname):
            yn = ask()
            while not (yn == 'y' or yn == 'n'):
                yn = ask()
            return yn == 'y'
        return True

    @classmethod
    def save(cls, obj, fname, force_overwrite = False):
        if not force_overwrite:
            # ask; if file already present and we need to overwrite
            if not Persist.overwrite(fname):
                print(f'Will not overwrite {fname}. Save failed!')
                return
        with open(fname, 'wb') as handle:
            pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, fname):
        with open(fname, 'rb') as handle:
            obj = pickle.load(handle)
        return obj
