from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class CLSP1:
    M: int = 3
    T: int = 6
    setup_costs: np.ndarray = np.array([100,120,110], dtype=np.float64)
    production_costs: np.ndarray = np.array([1,1,1], dtype=np.float64)
    inventory_costs: np.ndarray = np.array([3,2,1], dtype=np.float64)
    production_times: np.ndarray = np.array([1,1,1], dtype=np.float64)
    setup_times: np.ndarray = np.zeros(3, dtype=np.float64)
    capacities: np.ndarray = np.array([60,50,60,50,60,60], dtype=np.float64)
    demand: np.ndarray = np.array([[10,20,20,0,35,65],
                                   [20,0,3,18,13,65],
                                   [2,10,12,30,0,0]], dtype=np.float64)


@dataclass(frozen=True)
class CLSP2:
    M: int = 8
    T: int = 8
    setup_costs: np.ndarray = np.array([112,184,144,187,127,147,100,188], dtype=np.float64)
    production_costs: np.ndarray = np.array([1,1,1,1,1,1,1,1], dtype=np.float64)
    inventory_costs: np.ndarray = np.array([5,3,4,6,5,3,6,2], dtype=np.float64)
    production_times: np.ndarray = np.array([1,1,1,1,1,1,1,1], dtype=np.float64)
    setup_times: np.ndarray = np.array([2,5,5,1,8,3,6,2], dtype=np.float64)
    capacities: np.ndarray = np.array([200,200,210,200,160,190,170,170], dtype=np.float64)
    # each row corresponds to one product
    demand: np.ndarray = np.array([[43.0, 29.0, 52.0, 0.0, 0.0, 0.0, 42.0, 0.0],
       [30.0, 0.0, 0.0, 40.0, 20.0, 0.0, 6.0, 27.0],
       [0, 20, 0, 50, 60, 11, 0, 30],
       [33.0, 43.0, 30.0, 0.0, 16.0, 48.0, 37.0, 33.0],
       [0.0, 0.0, 0.0, 41.0, 16.0, 0.0, 55.0, 0.0],
       [0, 0, 21, 13, 7, 0, 22, 0],
       [0.0, 25.0, 43.0, 25.0, 0.0, 52.0, 10.0, 42.0],
       [42.0, 18.0, 40.0, 2.0, 0.0, 71.0, 0.0, 41.0]], dtype=np.float64)


@dataclass(frozen=True)
class CLSPL:
    M: int = 5
    T: int = 6
    setup_costs: np.ndarray = np.array([400,150,100, 100, 100], dtype=np.float64)
    production_costs: np.ndarray = np.array([0,0,0,0,0], dtype=np.float64)
    inventory_costs: np.ndarray = np.array([4,3,2,2,1], dtype=np.float64)
    production_times: np.ndarray = np.array([1,1,1,1,1], dtype=np.float64)
    setup_times: np.ndarray = np.array([10,10,10,10,10], dtype=np.float64)
    capacities: np.ndarray = np.array([200,200,200,200,200,200], dtype=np.float64)
    demand: np.ndarray = np.array([[30,0,80,0,40,0],
                                   [0,0,30,0,70,0],
                                   [0,0,40,0,60,0],
                                   [0,0,20,0,0,10],
                                   [0,0,60,0,50,0]], dtype=np.float64)

