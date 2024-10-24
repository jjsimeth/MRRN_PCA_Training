import multiprocessing as mp
import PCA_Training_SWI_rb as train
import platform as plat

if __name__ == '__main__':
    # Check what os we are running on.
    os_name = plat.system()

    # Enable multiprocessing if on windows
    if os_name == 'Windows':
        print(f" Detected os is: {os_name}. Freeze support is required to run the application")
        mp.freeze_support()
    else:
        print(f" Detected os is: {os_name}. Freeze support is not needed to run the application")
        pass

    # Start training
    train.train()