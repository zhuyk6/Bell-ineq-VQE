from dm import *
from qnn import *
from run import *

import json

def save_data(datas: dict, filename: str):
    with open(filename, "w") as f:
        json.dump(datas, f, cls=MyEncoder)


def randnew_q3(n: int, m: int):
    arr_s = []
    for i in range(n):
        dm = rand_density_matrix_from_basis(3, m)
        s = Svetlichny_qnn(dm)
        arr_s.append(s)
    
    save_data({"arr_s": arr_s}, f"randnew_q3_n{n}_m{m}.json")


if __name__ == "__main__":
    for m in range(1, 5):
        print(f"m = {m}")
        randnew_q3(1000, m)
