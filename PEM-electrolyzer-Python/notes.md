# Here are my notes for bugs fixing and todo list


## Todos


- [ ] Group soln algorthems ( Scalarization method/ Pareto-based Methods)
- [ ] User should have All constrainst in a range With slider
- [ ] Results: Pareto front in ovjective space + Pareto fornt in decision space
- [ ] Design table  with all vaiables and smulated smaple set
- [ ]Include measuring metics( hypervolume, C-metrics, eclidea distance)
- [ ] add dropdown notes, to explain reuslts metics.

- [ ] Results: Save temporal results to a folder and have a comparison and analysis page
- [ ]  Resulsts AI report page with template generation 
- 
- [ ] Proof reed model with litterature reference

-Always include units at least as comment code.

- [ ] check all constraiint with gemini

- [ ] -Add: Uncertianty analysis page

&#x2611 
:white_check_mark:
:heavy_check_mark:

### Solar System Exploration, 1950s – 1960s




## Error log and fixes


- error
from pymoo.factory import get_reference_directions  

"MOGA": MOGA(pop_size=pop_size)
 termination = get_termination("n_gen", n_gen)

3-from pymoo.algorithms.so.genetic_algorithm import GA
- fix

from pymoo.util.ref_dirs import get_reference_directions  
from pymoo.termination import get_termination
commented


3-In pymoo version 0.5.0 and later, the genetic_algorithm module has been moved to pymoo.algorithms.so.ga
from pymoo.algorithms.soo.nonconvex.ga import GA