##########################################################
import time
import numpy as np
import asyncio
from utils.utils import generateInputFileDict
from os.path import dirname


##########################################################
async def calculate(dirname=""):
    try:
        inputData = generateInputFileDict()
        print("Completed Calculation")
    except Exception as e:
        print("\nError Message: " + str(e) + "\n")
        return str(e)
    return inputData


##########################################################
async def main():
    ######################################################
    _ = await calculate(dirname=dirname(__file__))

##########################################################

######################################################
if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
    print("Done")
######################################################
