# Astro 528, Lab 6

Before starting this lab, make sure you've successfully gotten setup to use git, Julia and Pluto.
The first lab contained detailed instructions for using the Jupyter Lab server to work with Pluto notebooks.  

If you'll be using the ACI portal to access the JupyterLab server, then you need to request multiple processor cores when you first submit the request for the JupyterLab session using the box labeled "Number of Cores", i.e. before you start executing cells in this notebook.
![screen shot showing how to request multiple cores](images/portal_screenshot.png)

While we're in class (or the night before the lab is due), please ask for just 4 cores, since there are likely nearly ~12 of us using the system at once.
When working on the lab (or your project) outside of class, feel free to try benchmarking the code using 8 cores or even 16 cores.  For sessions where you ask for several cores, then please be extra diligent about closing your session when you're done, so they are made available for other users.

Remember, that you need follow the provided link to create your own private copy of this lab's repository on GitHub.com.   See the
[help on the course website](https://psuastro528.github.io/tips/submitting/) for instructions on cloning, committing, pushing and reviewing your work.


## Exercise 1:  Parallelization for Multi-Core Workstations via Multiple-Threads
### Goals:
- Choose an appropriate number of worker processors for your compute node
- Parallelize code using ThreadsX.map and ThreadsX.mapreduce
- Parallelize code using Threads.@threads
- Parallelize code using ThreadsX.foreach
- Parallelize code using FLoops & ThreadedEx

From a Pluto session, work through ex1.jl


## Exercise 2:  Parallelization for Multi-Core Workstations via Multiple Processes
### Goals:
- Load code and packages on worker nodes
- Parallelize code using pmap
- Parallelize code using SharedArray's
- Parallelize code using map and mapreduce on DistributedArray's
- Parallelize code using @distributed for loop
- Parallelize code using @distributed for loop with a reducing operation
- Parallelize code using FLoops & DistributedEx

From a **Jupyter** session, work through ex2.ipynb.
Once you've finished your lab, remember to [create, commit and push a markdown version of your Jupyter notebook](http://localhost:1313/tips/submitting/submitting/#convert-to-markdown).  The short version is
```bash
julia --project=. -e 'using Weave; convert_doc("NOTEBOOK_NAME.ipynb","NOTEBOOK_NAME.jmd")'

git add NOTEBOOK_NAME.jmd  
git commit -m "Adding markdown version of Jupyternotebok"
git push
```
