### A Pluto.jl notebook ###
# v0.19.27

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 76730d06-06da-4466-8814-2096b221090f
begin
	# Packages for Notebook experience
	using PlutoUI, PlutoTeachingTools, PlutoTest
	using Plots

	# Packages for parallelization
	#using SharedArrays
	using CpuId
	using ThreadsX
	using FLoops
	# Packages for benchmarking
	using BenchmarkTools

	# Packages needed by model
	using Distributions, Random
	using QuadGK
	using StaticArrays
	Random.seed!(42)

	nb_link_prefix = PlutoRunner.notebook_id[] |>string; # for making urls to notebook
end

# ╔═╡ 85aad005-eac0-4f71-a32c-c8361c31813b
md"""
# Lab 6, Exercise 1
## Parallelization: Shared-memory model, Multi-threading
"""

# ╔═╡ bdf61711-36e0-40d5-b0c5-3bac20a25aa3
md"""
In this lab, we'll explore a multiple different ways that we can parallelize calculations across multiple cores of a single workstation or server.
This exercise will focus on parallelization using multiple *threads*.
A separate exercise will focus on parallelization using multiple *processes*, but using a Jupyter notebook, rather than a Pluto notebook (due to internals of how Pluto works).
"""

# ╔═╡ 629442ba-a968-4e35-a7cb-d42a0a8783b4
protip(md"""
In my experience, parallelization via multiple threads tends to be more efficient than using multiple processes.  Multi-threading is my "go-to" method for an initial parallelization.  That said, it's good to be aware of some of the reasons that others may choose to parallelize their code over multiple processes (e.g., if you're concerned about security of data, robustness to errors in one process).  For me, the main advantage of using multiple processes is that multiple processes will be necessary once we transition to distributed memory computing.  Therefore, parallelizing your code using multiple processes can make it easier to scale up to more cores than are avaliable in a single node.

That said, near the end of this exercise we'll see an example of how a programming interfaces that makes it easy to transition code between multi-threaded and mulit-process models.
""")

# ╔═╡ 0bee1c3c-b130-49f2-baa4-efd8e3b49fdc
md"""
## Hardware & Pluto server configuration
Most modern workstations and even laptops have multiple processor cores.
If you're using the Roar Collab portal and BYOE JupyterLab server, then you need to request that multiple processor cores be allocated to your session when you first submit the request for the BYOE JupyterLab server using the box labeled "Number of Cores", i.e. before you open this notebook and even before you start your Pluto session.
"""

# ╔═╡ f76f329a-8dde-4790-96f2-ade735643aeb
if haskey(ENV,"PBS_NUM_PPN")
	procs_per_node = parse(Int64,ENV["PBS_NUM_PPN"])
	md"Your PBS job was allocated $procs_per_node CPU cores per node."
elseif haskey(ENV,"SLURM_CPUS_PER_TASK") && haskey(ENV,"SLURM_TASKS_PER_NODE")
    procs_per_task = parse(Int64,ENV["SLURM_CPUS_PER_TASK"])
    tasks_per_node = parse(Int64,ENV["SLURM_TASKS_PER_NODE"])
	procs_per_node = procs_per_task * tasks_per_node
	md"Your Slurm job was allocated $procs_per_node CPU cores per node."
else
	procs_per_node = missing
	md"It appears you're not running this on Roar Collab.  Later in the notebook, we'll try to use all the cores on your local machine."
end

# ╔═╡ 0e4d7808-47e2-4740-ab93-5d3973eecaa8
if !ismissing(procs_per_node)
	if procs_per_node > 4
		warning_box(md"""While we're in class (and the afternoon/evening before labs are due), please ask for just 4 cores, so there will be enough to go around.

		If you return to working on the lab outside of class, then feel free to try benchmarking the code using 8 cores or even 16 cores. Anytime you ask for several cores, then please be extra diligent about closing your session when you're done.""")
	end		
end

# ╔═╡ 8a50e9fa-031c-4912-8a2d-466e6a9a9935
md"""
This notebook is using **$(Threads.nthreads()) threads**.
"""

# ╔═╡ 7df5fc86-889f-4a5e-ac2b-8c6f68d7c32e
warning_box(md"""
Even when you have a JupyterLab server (or remote desktop or Slurm or PBS job) that has been allocated multiple CPU cores, that doesn't mean that any code will make use of more than one core.  The Roar Collab Portal's Pluto server for this class has been configured to start notebooks with as many threads as physical cores that were allocated to the parent job.

If you start julia manually (e.g., from the command line or remote desktop), then you should check that its using the desired number of threads.  The number can be can control using either the `JULIA_NUM_THREADS` environment variable or the `-t` option on the command line.  Somewhat confusingly, even if you start julia using multiple threads, that doesn't mean that the Pluto server will assign that many threads to each notebook.  If you run your own Pluto server, then you can control the number of threads used within a notebook by starting it with
```julia
using Pluto
Pluto.run(threads=4)
```""")

# ╔═╡ 571cab3f-771e-4464-959e-f351194049e2
md"""
Before we get started, let's get some information about the processor that our server is running on and double check that we're set to use an appropriate number of threads.
"""

# ╔═╡ 0c775b35-702e-4664-bd23-7557e4e189f4
with_terminal() do
	Sys.cpu_summary()
end

# ╔═╡ 3059f3c2-cabf-4e20-adaa-9b6d0c07184f
md"""
If you're running this notebook on your own computer, then we'll want to make sure that we set the number of threads to be no more than the number of processor cores listed above. It's very likely that you might be better off requesting only half the number of processors as listed above. (Many processors present themselves as having more cores than they actually do. For some applications, this can be useful.  For many scientific applications it's better to only use as many threads as physical cores that are avaliable.
"""

# ╔═╡ 4fa907d0-c556-45df-8056-72041edcf430
md"""
The [CpuId.jl](https://github.com/m-j-w/CpuId.jl) package provides some useful functions to query the properties of the processor you're running on.
"""

# ╔═╡ 73e5e40a-1e59-41ed-a48d-7fb99f5a6755
cpucores()   # query number of physical cores

# ╔═╡ f97f1815-50a2-46a9-ac20-e4a3e34d898c
cputhreads() # query number of logical cores

# ╔═╡ 53da8d7a-8620-4fe5-81ba-f615d2d4ed2a
if cpucores() < cputhreads()
	warning_box(md"""Your processor is presenting itself as having $(cputhreads()) cores, when it really only has $(cpucores()) cores.  Make sure to limit the number of threads to $(cpucores()).  
	
	If you're running on Roar Collab, then you should also limit the number of threads you use to the number of CPU cores assigned to your job by the slurm workload manager.
	""")
end

# ╔═╡ cc1418c8-3261-4c70-bc19-2921695570a6
Threads.nthreads()  # Number of threads avaliable to this Pluto notebook

# ╔═╡ 7f724449-e90e-4f8b-b13c-9640a498893c
@test 1 <= Threads.nthreads() <= cpucores()

# ╔═╡ c85e51b2-2d3d-46a2-8f3f-03b289cab288
 @test !ismissing(procs_per_node) && 1 <= Threads.nthreads() <= procs_per_node

# ╔═╡ 907766c5-f084-4ddc-bb52-336cb037d521
md"1a.  How many threads is your notebook using?  (Please enter it as an integer rather than a function call, so that it gets stored in your notebook.  That way the TA and I will be able to interpret the speed-up factors you get below.)"

# ╔═╡ 0bcde4df-1e31-4774-a31f-bd451bb6f758
response_1a = 8 # missing # Insert response as simple integer, and not as a variable for function

# ╔═╡ c41d65e3-ea35-4f97-90a1-bfeaeaf927ad
begin
    if !@isdefined(response_1a)
		var_not_defined(:response_1a)
    elseif ismissing(response_1a)
    	still_missing()
	elseif !(typeof(response_1a) <: Integer)
		warning_box(md"response_1a should be an Integer")
	elseif !(1<(response_1a))
		warning_box(md"Please restart your JupyterLab session and use at least 2 cores.")
	elseif (response_1a) != Threads.nthreads()
		warning_box(md"That's not what I was expecting.  Please double check your response.")
	else
		correct(md"Thank you.")
	end
end


# ╔═╡ 6e617a7c-a640-4cb3-9451-28a0036d8fdc
md"# Calculation to parallelize"

# ╔═╡ 5e6c430a-cd2f-4169-a5c7-a92acef813ac
md"""
For this lab, I've written several functions that will be used to generate simulated spectra with multiple absorption lines.  This serves a couple of purposes.
First, you'll use the code in the exercise, so you have a calculation that's big enough to be worth parallelizing.  For the purposes of this exercise, it's not essential that you review the code I provided in the `src/*.jl` files.  However, the second purpose of this example is providing code that demonstrates several of the programming patterns that we've discussed in class.  For example, the code in the `ModelSpectrum` module
- is in the form of several small functions, each which does one specific task.
- has been moved out of the Jupyter notebook and into `.jl` files in the `src` directory.
- creates objects that compute a model spectrum and a convolution kernel.
- uses [abstract types](https://docs.julialang.org/en/v1/manual/types/#Abstract-Types-1) and [parametric types](https://docs.julialang.org/en/v1/manual/types/#Parametric-Types-1), so as to create type-stable functions.
- has been put into a Julia [module](https://docs.julialang.org/en/v1/manual/modules/index.html), so that it can be easily loaded and so as to limit potential for namespace conflicts.

You don't need to read all of this code right now.  But, when you're writing code for your class project, you're likely to want to make use of some of these same programming patterns.   It may be useful to refer back to this code later to help see examples of how to apply these design patterns in practice.

In the Helper code section at the bottom of the notebook, we read the code in `src/model_spectrum.jl` and place it in a module named ModelSpectrum.  Note that this implicitly includes the code from other files: `continuum.jl`, `spectrum.jl` and `convolution_kernels.jl`.
Then we'll bring several of the custom types into scope, so we can use them easily below.
"""

# ╔═╡ c31cf36c-21ec-46f1-96aa-b014ff094f8a
md"""
## Synthetic Spectrum
In this exercise, we're going to create a model spectrum consisting of continuum, stellar absorption lines, telluric absorption lines.
The `ModelSpectrum` module provides a `SimulatedSpectrum` type.
We need to create a `SimulatedSpectrum` object that contains specific parameter values.  The function below will do that for us.
"""

# ╔═╡ 7026e51d-c3e4-4503-9f35-71074b0c2f1a
md"""
Next, we specify a set of wavelengths where the spectrum will be defined,
and create a functor (or function-like object) that contains all the line properties and can compute the synethic spectrum.
"""

# ╔═╡ ad302f2b-69dc-4559-ba12-d7fb2e8e689e
begin  # Pick range of of wavelength to work on.
	lambda_min = 5000
	lambda_max = 6000
end;

# ╔═╡ 16ad0225-c7d6-455b-8eb0-3e93c9f9f91a
md"## Convolved spectrum

Next, we will create an object containing a model for the point spread function (implemented as a mixture of multiple Gaussians).
Then we create a funtor that can compute the convolution of our spectral model with the point spread function model.
"

# ╔═╡ 324a9a25-1ec4-4dc2-a7ca-e0f1f56dbf66
md"""
## Visualize the models
Before going further, it's probably useful to plot both the raw spectrum and the convolved spectrum.
"""

# ╔═╡ 52127f57-9a07-451a-bb24-c1f3c5581f0a
begin 	# You may want to adjust the num_lambda to make things more/less computationally intensive
	num_lambda = 4*1024
	lambdas = range(lambda_min,stop=lambda_max, length=num_lambda)
	lambdas = collect(lambdas) # to make an actual array
end;

# ╔═╡ 75948469-1347-45e2-9281-f366b41d0e04
md"""
That's fairly crowded, you it may be useful to zoom in on a narrower range.
"""

# ╔═╡ 4d1cf57f-b394-4f37-98c3-0d765f4ee635
md"""
Plot width:
$(@bind idx_plt_width Slider(8:min(1024,length(lambdas)), default=min(128,floor(Int,length(lambdas)//2)) ) )
center:
  $(@bind idx_plt_center Slider(1:length(lambdas), default = floor(Int,length(lambdas)//2)) )

"""

# ╔═╡ cddd761a-f051-4338-9e40-d35e050060d3
begin
		idx_plt_lo = max(1,idx_plt_center - idx_plt_width)
		idx_plt_hi = min(length(lambdas),idx_plt_center + idx_plt_width)
		idx_plot = idx_plt_lo:idx_plt_hi
end;

# ╔═╡ ee96411d-e3fa-442b-b0fe-10d6ede37b6a
md"""
You can adjust the sliders to interactively explore our model spectra.
"""

# ╔═╡ b92aad2e-8a3b-4edf-ae7e-6e3cff6eead4
protip(md"Feel free to look at the hidden code in the cells above for the lower plot and slider bars, as well as the documentation at [PlutoUI.jl](https://docs.juliahub.com/PlutoUI/abXFp/0.7.52/) or the [example notebook](https://featured.plutojl.org/basic/plutoui.jl) for examples of how to make interactive widgets in your notebooks.")

# ╔═╡ e5f9fa06-9fbb-40a8-92de-71523775d257
md"""
# Serial implementations
## Benchmarking spectrum (w/o convolution)

Before we parallelize anything, we want to benchmark the calculation of spectra on a single processor.  To avoid an annoying lag when using the notebook, we won't use the `@benchmark` script.  Instead, we'll run each calculation just twice, once to ensure it's compiled and a second time for benchmarking it with the `@timed` macro.  When it comes time to benchmark your project code, you'll want to collect multiple samples to get accurate benchmarking results.
"""

# ╔═╡ b195ebd2-9584-40b8-ae3e-6d9ce88b5398
md"""
Let's think about what's happening with the serial version.
With `raw_spectrum(lambdas)` or `raw_spectrum.(lambdas)` we will evalute the spectrum model at each of the specified wavelengths using a few different syntaxes.
"""

# ╔═╡ d6d3a2d1-241e-44c1-a11b-5bfb2b3c5f4b
md"""
As expected, the different versions perform very similarly in terms of wall-clock time and memory allocated.
"""

# ╔═╡ 0344a74d-456b-44f0-84dc-c2fdbd41a379
md"""
## Benchmarking convolved spectrum

Next, we'll evaluate the convolution of the raw spectrum with the PDF model at each of the wavelengths, using `conv_spectrum`.
"""

# ╔═╡ 51adffd7-8fb6-4ed2-8510-303a37d6efc3
md"""
Now, the two implementations performed very differently.  Let's think about what's causing that difference.
In each case, the convolution integral is being computed numerically by [QuadGK.jl](https://github.com/JuliaMath/QuadGK.jl).  On one hand, it's impressive that QuadGK.jl was written in a generic way, so that it can compute an integral of a scalar (when we used the broadcasting notation) or integral of vectors (when we passed the vector of wavelengths without broadcasting).
On the other hand, there's a significant difference in the wall clock time and lots more memory being allocated when we pass the vector, instead of using broadcasting.
When we pass a vector, the `quadgk` is computing the convolution integral is using vectors.  Since the size of the vectors isn't known at compile time they must be allocated  on the heap.  This results in many unnecessary memory allocations (compared to if the calculations were done one wavelength at a time).

We can get around this problem by using broadcasting or map, so the convolution integral is performed on scalars, once for each wavelength.  This significantly reduces the number of memory allocations and the runtime.  This also has the advantage that we've broken up the work into many independent calculations that could be performed in parallel.
"""

# ╔═╡ 71d943e3-761a-4337-b412-b0b768483bc2
protip(md"Interestingly, there's actually more work to do in the case of computing integrals of scalars, since the adaptive quadrature algorithm chooses how many points and and where to evaluate the integrand separately for each wavelength.  However, the added cost of memory allocations is much more expensive than the cost of the added calculations.

Another complicating factor, the answers aren't identical.  This is because the criteria used by `quadgk` for when to stop evaluating the integrand at more points changes depending on whether it's deciding when to stop for each wavelength separately or for the entire vector at once.

In principle, we could further optimize the serial version to avoid unnecessary memory allocations.  QuadGK.jl provides a function `quadgk!` that writes the output into a preallocated space.  Even `quadgk!` needs some memory to compute intermediate values.  Normally,  `quadgk` or `quadgk!` will allocate a buffer for segments automatically.  However, you can instead allocate a buffer using `alloc_segbuf(...)` and pass the preallocated buffer as the `segbuf` argument.  When using multiple threads, we'd need to allocate a separate buffer for each thread and make sure that each thread uses only its own buffer.  However, it would take some time to figure out how to do that and to test the resulting code.  In practice, it's often a better use of our time make a pretty good serial code that can be parallelized well and to use of our time to parallelize that, rather than making most efficient serial code possible.")

# ╔═╡ db1583f4-61cb-43e0-9326-d6c15d8fad5a
md"""
## Map
Our calculation is one example of a very useful programming pattern, known as **map**.  The map pattern corresponds to problems where the total work can be organized as doing one smaller calculation many times with different input values.
Julia provides a [`map`](https://docs.julialang.org/en/v1/base/collections/#Base.map) function (as well as `map!` for writing to memory that's been preallocated ) that can be quite useful.
`map(func,collection)` applies func to every element of the collection and returns a collection similar in size to collection.
In our example, each input wavelength is mapped to our output flux.
"""

# ╔═╡ f108d26b-6c75-4eb6-9e88-a60ec038a73c
md"""
As expected, the map versions perform very similarly in terms of wall-clock time and memory allocated to the broadcasted versions for both the raw and convolved spectra.
"""

# ╔═╡ e71cede9-382e-47e2-953a-2fa96ed50002
md"## Loop (serial)"

# ╔═╡ 4d54b6a7-3fc0-4c63-8a9d-d683aa4ecefe
md"""
Sometimes it's cumbersome to write code in terms of `map` functions.  For example, you might be computing multiple quantities during one pass of your data (e.g., calculating a sample variance in lab 1).  In these cases, it's often more natural to write your code as a `for` loop.
"""

# ╔═╡ 21f305db-24e1-47d1-b1f4-be04ca91780e
protip(md"""
It is possible to have each function return an array.  Then the output is an array of arrays.  In that case we could use `stack` to return a 2-d array. 

However, if each function returns a NamedTuple (or a custom struct), then then the output of `map` is an array of NamedTuples (or an array of structs).  However, getting the outputs in the format we want to use for subsequent calculations (e.g., arrays for each output, rather than an array of structs) is often more tedious and error prone than just writing our code in terms of either a `for` loop or a broadcasted function.""")

# ╔═╡ a44a3478-541d-40d6-9d99-04b918c16bfb
md"""We'll implement a serial version as a starting point and comparison.
"""

# ╔═╡ 96914ff8-56c8-4cc8-96bc-fd3d13f7e4ce
md"As expected the performance is very similar to the broadcasted for mappeed version."

# ╔═╡ 32685a28-54d9-4c0d-8940-e82843d2cab2
md"# Parallelization via multiple threads"

# ╔═╡ 3717d201-0bc3-4e3c-8ecd-d835e58f6821
md"""
Julia has native support for using multiple **threads**.  This is useful when you have one computer with multiple processor cores.  Then each thread can execute on a separate processor core.  Because the threads are part of the same **process**, every thread has access to all the memory used by every other thread.  Programming with threads requires being careful to avoid undefined behavior because threads read and write to the same memory location in an unexpected order.  In general, multi-threaded programming can be intimidating, since arbitrary parallel code is hard to write, read, debug and maintain.  One way to keep things managable is to stick with some common programming patterns which are relatively easy to work with.  We'll explore using threads for a parallel for and a parallel map.
"""

# ╔═╡ 496e8c5e-251b-4448-8c59-541877d752c1
md"""
## Parallel Map

If you can write your computations in terms of calling `map`, then one easy way to parallelize your code is to replace the call to `map` with a call to `ThreadsX.map`, a parallel map that makes use of multiple threads.
If your julia kernel has only a single thread, then it will still run in serial.  But if you have multiple theads, then `ThreadsX.map` will parallelize your code.
"""

# ╔═╡ 04bcafcd-1d2f-4ce5-893f-7ec5bb05f9ed
md"""
1a.  Given that this notebook is using $(Threads.nthreads()) threads, what is the theoretical maximum improvement in performance?  How much faster do you expect the `conv_spectrum` code to run using `ThreadsX.map` relative to serial `map`?
"""

# ╔═╡ ca8ceb27-86ea-4b90-a1ae-86d794c9fc98
response_1b = missing  # md"Insert your responce"

# ╔═╡ 4ad081a2-b5c2-48ff-9a28-ec9c8d9f0d0e
begin
    if !@isdefined(response_1b)
		var_not_defined(:response_1b)
    elseif ismissing(response_1b)
    	still_missing()
	end
end

# ╔═╡ 2399ce76-b6da-4a61-bcda-aee22dd275f8
md"""
1c. How did the performance improvement compare to the theoretical maximum speed-up factor and your expectations?
"""

# ╔═╡ a25c6705-54f4-4bad-966e-a8f13ae4c711
response_1c = missing  # md"Insert your responce"

# ╔═╡ 739136b1-6b01-44c0-bbfd-dcb490d1e191
begin
    if !@isdefined(response_1c)
		var_not_defined(:response_1c)
    elseif ismissing(response_1c)
    	still_missing()
	end
end

# ╔═╡ dcce9a84-a9b1-47c1-8e08-7575cb299b56
md"""
You were likely a little disappointed in the speed-up factor.  What could have gone wrong?  In this case, we have a non-trivial, but still modest amount of work to do for each wavelength.  `map` distributed the work one element at a time.  The overhead in distributing the work and assembling the pieces likely ate into the potential performance gains.  To improve on this, we can tell `map` to distribute the work in batches.  Below, we'll specify an optional named parameter, `basesize`.  (Feel free to try chaning the size of batches to see how that affects the runtime.)
"""

# ╔═╡ fb063bc5-22bc-4b32-8fcb-5fbc4765c8b5
batchsize_for_ThreadsXmap = 256

# ╔═╡ 90c9d079-4bbc-4609-aa12-afa41a74b2fb
md"""
1d.  After specifying a batchsize, how much faster was the code using `ThreadsX.map` with batches than the the serial version?  How does this compare to the theoretical maximum speed-up factor and your original expectations? 
"""

# ╔═╡ 0edbb2db-4db8-4dc4-9a73-f7ff86e6f577
response_1d = missing  # md"Insert your responce"

# ╔═╡ a944fdea-f41b-4a5f-95ac-e5f4074d4290
begin
    if !@isdefined(response_1d)
		var_not_defined(:response_1d)
    elseif ismissing(response_1d)
    	still_missing()
	end
end

# ╔═╡ bd81357b-c461-458e-801c-610893dd5ea1
md"## Parallel Loop"

# ╔═╡ 0e0c25d4-35b9-429b-8223-90e9e8be90f9
md"""
It is also possible to parallelize for loops using multiple threads.  Julia's built-in `Threads` module provides one implementation.
"""

# ╔═╡ 5a63ebd6-3e18-49ee-8d1d-4bb2da6419b6
md"""
1e.  How much faster do you expect the `conv_spectrum` code to run using `Threads.@threads for...` relative to searial `for`?
"""

# ╔═╡ 86e7d984-c128-4d2e-8599-3bc70db87a1d
response_1e = missing # md"Insert your response"

# ╔═╡ c69c0a4a-b90b-414c-883d-3aa50c04b5e1
begin
    if !@isdefined(response_1e)
		var_not_defined(:response_1e)
    elseif ismissing(response_1e)
    	still_missing()
	end
end

# ╔═╡ 791041e9-d277-4cac-a5ac-1d6ec52e0287
md"""
While Threads.@threads can be useful for some simple tasks, there is active development of packages that provide additional features for multi-threaded programming.  For example, the ThreadsX package provides a `foreach` function and the FLoops package provides a `@floop` macro, both of which we'll demonstrate and benchmark below.
"""

# ╔═╡ 7c367a0b-c5b9-459b-9ccf-e07c84e0b32a
protip(md"""
There are several more packages to help you parallel code efficiently in different circumstances.  For example, [ThreadPools](https://github.com/tro3/ThreadPools.jl) provides multiple variants of `map` and `foreach`, so you can easily choose between how work is scheduled among the workers and whether the delegator thread is assigned work.  
""")

# ╔═╡ 2b00f6fc-9bfd-48d6-a4d8-ac95f7e71faa
md"""
Inevitably, one package/pattern for parallelizing your code will be a little more efficient than the others. But there are often multiple ways to implement parallelism that are comparable in run-time. When the performance is similar, other considerations (e.g., ease of programming, quality of documentation, ease swapping out different parallelization strategies) may play a major role in your decision of how to implement parallelism.
"""

# ╔═╡ ea002e89-9f4e-441e-8998-5e9c99bb27e0
md"""
At first, it may seem like the above examples are just alternative syntaxes for writing a loop parallelized over multiple threads.  Why are these worth learning about?  

ThreadsX provides a drop-in replacement for several functions from Base.  The common interface makes it easy to swap in for serial code quickly.  

While FLoops requires a somewhat different syntax, it makes it relatively easy to swap between multiple forms of parallelism.  Therefore, writing your code so it can be multi-threaded using FLoops is likely to make it very easy to parallelize your code for a distributed memory architecture.  FLoops can even make it easy to parallelize codes using a GPU.  Thus, it's worth keeping these in mind when planning your project.  
"""

# ╔═╡ d43525da-e0a2-4d2f-9dbb-bf187eebf6c1
tip(md"""
## ''Embarassingly'' parallel is good

So far, we've demonstrated parallelizing a computation that can be easily broken into smaller tasks that do not need to communicate with each other.  This is often called an called *embarassingly parallel* computation.  Don't let the name mislead you.  While it could be embarassingly if a Computer Science graduate student tried to make a Ph.D. thesis out of parallelizing an embarassingly parallel problem, that doesn't mean that programmers shouldn't take advantage of opportunities to use embarssingly parallel techniques when they can.  If you can parallelize your code using embarassingly parallel techniques, then you should almost always parallelize it that way, instead of (or at least before) trying to parallelize it at a finer grained level.

Next, we'll consider problems that do require some communications between tasks, but in a very structured manner.
""")

# ╔═╡ 547ad5ba-06ad-4707-a7ef-e444cf88ae53
md"""
# Reductions
Many common calculations can be formulated as a [**reduction operation**](https://en.wikipedia.org/wiki/Reduction_operator), where many inputs are transformed into one output.  Common examples would be `sum` or `maximum`.  One key property of reduction operations is that they are associative, meaning it's ok for the computer to change the order in which inputs are reduced.  (Thinking back to our lesson about floating point arithmetic, many operations aren't formally associative or commutative, but are still close enough that we're willing to let the computer reorder calculations.)

When we have multiple processors, the input can be divided into subsets and each processor reduce each subset separately.  Then each processor only needs to communicate one value of the variable being reduced to another processor, even if the input is quite large.  For some problems, reductions also reduce the amount of memory allocations necessary.
"""

# ╔═╡ 7ba35a63-ac61-434b-b759-95d505f62d9e
md"""
We'll explore different ways to perform reductions on an example problem where we calculate the mean squared error between the model and the model Doppler shifted by a velocity, $v$. First, let's write a vanilla serial version, where we first compute an array of squared residuals and pass that to the `sum` function.
"""

# ╔═╡ cee9c93d-cf7b-4da1-b4bb-b544b7cc104c
v = 10.0

# ╔═╡ 3ac01c04-52e3-497e-8c29-8c704e23ae39
md"## Serial loop with reduction"

# ╔═╡ 790377a7-1301-44a8-b300-418567737373
md"""
Now we'll write a version of the function using a serial for loop.  Note that we no longer need to allocate an output array, since `calc_mse_loop` only needs to return the reduced mean squared error and not the value of the spectrum at every wavelength.
"""

# ╔═╡ 161ea6af-5661-44e1-ae40-1b581b636c25
md"""
## Parallel loop with reduction
Next, we'll use [FLoops.jl](https://github.com/JuliaFolds/FLoops.jl) to compute the mean sequared error using multiple threads.  Note that we need to use the `@floop` macro around the loop  *and* the `@reduce` macro to indicate which variables are part of the reduction.
"""

# ╔═╡ 7def3535-6f90-4bf8-b86f-aac278666663
md"""
1f.  How do you expect the performance of `calc_mse_flloop` to compare to the performance of `calc_spectrum_flloop` and `calc_mse_loop`?
"""

# ╔═╡ 1989da2a-1fe2-49a0-b279-5925ae4b428c
response_1f = missing # md"Insert your response"

# ╔═╡ 8d7c27d5-4a07-4ab4-9ece-94fdb7053f73
begin
    if !@isdefined(response_1f)
		var_not_defined(:response_1f)
    elseif ismissing(response_1f)
    	still_missing()
	end
end

# ╔═╡ 8e9b1e02-2bc0-49d2-b7ed-38de877ebe77
response_1g = missing # md"Insert your response"

# ╔═╡ ba62f716-b1b5-4d11-91f2-ed121b48216c
begin
    if !@isdefined(response_1g)
		var_not_defined(:response_1g)
    elseif ismissing(response_1g)
    	still_missing()
	end
end

# ╔═╡ bbdd495c-f2c6-4264-a4e9-5083753eb410
md"""
One advantage of parallelizing your code with [FLoops.jl](https://juliafolds.github.io/FLoops.jl/dev/) is that it then becomes very easy to compare the performance of a calculation in serial and in parallel using different **[executors](https://juliafolds.github.io/FLoops.jl/dev/tutorials/parallel/#tutorials-executor)** that specify how the calculation should be implemented.  There are different parallel executor for shared-memory parallelism (via multi-threading this exercise), distributed-memory parallelism (see [Lab 7](https://github.com/PsuAstro528/lab7-start)) and even for parallelizing code over a GPUs (although there are some restrictions on what code can be run on the GPU, that we'll see in a [Lab 8](https://github.com/PsuAstro528/lab8-start)).
"""

# ╔═╡ 383aa611-e115-482e-873c-4487e53d457f
md"# Mapreduce

We can combine `map` and `reduce` into one function `mapreduce`.  There are opportunities for some increased efficiencies when merging the two, since the amount of communications between threads can be significantly decreased thanks to the reduction operator.  Mapreduce is a common, powerful and efficient programming pattern.  For example, we often want to evaluate a model for many input values, compare the results of the model to data and the compute some statistic about how much the model and data differ.

In this exercise, we'll demonstrate using `mapreduce` for calculating the mean squared error between the model and the model Doppler shifted by a velocity, $v$.  First, we'll
"

# ╔═╡ 2c6fa743-3dec-417b-b05a-17bb52b5d39d
 md"## Mapreduce (serial)"

# ╔═╡ ae47ef38-e8d0-40b9-9e61-3ab3ca7e7a49
md"## Parallel mapreduce"

# ╔═╡ aad94861-e2b3-417d-b640-b821e53adb23
md"""
The ThreadsX package provides a multi-threaded version of mapreduce that we can easily drop in.
"""

# ╔═╡ f1c0321b-7811-42b1-9d0c-9c69f43d7e1a
md"""
Similar to before, we may be able to reduce the overhead associated with distributing work across threads by grouping the calculations into batches.  
"""

# ╔═╡ df044a68-605f-4347-832a-68090ee07950
mapreduce_batchsize = 8

# ╔═╡ 3f01d534-b01d-4ab4-b3cd-e809b02563a9
md"""
1h.  How did the performance of `calc_mse_mapreduce_threadsx` compare to the performance of `calc_mse_map_mapreduce`?  Can you explain why this differs from the comparison of `calc_spectrum_mapreduce_threadsx` to `ThreadsX.map(conv_spectrum,lambdas,..)`?
"""

# ╔═╡ d16adf94-72c3-480d-bd92-738e806068f8
response_1h = missing # md"Insert your response"

# ╔═╡ 56c5b496-a063-459a-8686-22fc70b6a214
begin
    if !@isdefined(response_1h)
		var_not_defined(:response_1h)
    elseif ismissing(response_1h)
    	still_missing()
	end
end

# ╔═╡ c4ff4add-ab3c-4585-900e-41f17e905ac5
md"""
1i.  Think about how you will parallelize your class project code.  The first parallelization typically uses a shared-memory model.  Which of these programming patterns would be a good fit for your project?  Can your project calculation be formulated as a `map` or `mapreduce` problem?  If not, then could it be implemented as a series of multiple maps/reductions/mapreduces?

Which of the parallel programming strategies are well-suited for your project?

After having worked through this lab, do you anticipate any barriers to applying one of these techniques to your project?

"""

# ╔═╡ ac18f1ca-0f60-4436-9d8a-797b3dfd8657
response_1i = missing  #= md"""
Insert your
multi-line
response
"""
=#

# ╔═╡ e8082779-143d-4562-81f3-d493679cf3c7
begin
    if !@isdefined(response_1i)
		var_not_defined(:response_1i)
    elseif ismissing(response_1i)
    	still_missing()
	end
end

# ╔═╡ bd77bc71-ffdf-4ba1-b1ee-6f2a69044e6f
begin
    σ_obs1 = 0.02*ones(size(lambdas))
    σ_obs2 = 0.02*ones(size(lambdas))
end;

# ╔═╡ 3b50062c-99c1-4f68-aabe-2d40d4ad7504
md"## Helper code"

# ╔═╡ d83a282e-cb2b-4837-bfd4-8404b3722e3a
ChooseDisplayMode()

# ╔═╡ c9cf6fb3-0146-42e6-aaae-24e97254c805
TableOfContents(aside=true)

# ╔═╡ 73358bcf-4129-46be-bef4-f623b11e245b
begin
	# Code for our model
	ModelSpectrum = @ingredients "./src/model_spectrum.jl"
	import .ModelSpectrum:AbstractSpectrum, SimulatedSpectrum, ConvolvedSpectrum, GaussianMixtureConvolutionKernel, doppler_shifted_spectrum
end

# ╔═╡ 4effbde2-2764-4c51-a9d0-a2db82f60862
"Create an object that provides a model for the raw spetrum (i.e., before entering the telescope)"
function make_spectrum_object(;lambda_min = 4500, lambda_max = 7500, flux_scale = 1.0,
        num_star_lines = 200, num_telluric_lines = 100, limit_line_effect = 10.0)

    continuum_param = flux_scale .* [1.0, 1e-5, -2e-8]

    star_line_locs = rand(Uniform(lambda_min,lambda_max),num_star_lines)
    star_line_widths = fill(1.0,num_star_lines)
    star_line_depths = rand(Uniform(0,1.0),num_star_lines)

    telluric_line_locs = rand(Uniform(lambda_min,lambda_max),num_telluric_lines)
    telluric_line_widths = fill(0.2,num_telluric_lines)
    telluric_line_depths = rand(Uniform(0,0.4),num_telluric_lines)

	SimulatedSpectrum(star_line_locs,star_line_widths,star_line_depths,telluric_line_locs,telluric_line_widths,telluric_line_depths,continuum_param=continuum_param,lambda_mid=0.5*(lambda_min+lambda_max),limit_line_effect=limit_line_effect)
end

# ╔═╡ 86b8dd31-1261-4fb9-bfd3-13f6f01e7790
# Create a functor (function object) that computes a model spectrum that we'll analyze below
raw_spectrum = make_spectrum_object(lambda_min=lambda_min,lambda_max=lambda_max)

# ╔═╡ 658f73c3-1e7a-47da-9130-06673f484ba1
if true
	raw_spectrum(lambdas)
	stats_serial_raw = @timed raw_spectrum(lambdas)
	(;  time=stats_serial_raw.time, bytes=stats_serial_raw.bytes)
end

# ╔═╡ 1c069610-4468-4d10-98f7-99662c26bdda
if true
	raw_spectrum.(lambdas)
	stats_broadcasted_serial_raw = @timed raw_spectrum.(lambdas)
	(;  time=stats_broadcasted_serial_raw.time, bytes=stats_broadcasted_serial_raw.bytes)
end

# ╔═╡ ca9c7d9e-e6cc-46cc-8a9b-ccda123591a2
if true
	map(raw_spectrum,lambdas)
	stats_map_serial_raw = @timed map(raw_spectrum,lambdas)
	(;  time=stats_map_serial_raw.time, bytes=stats_map_serial_raw.bytes)
end

# ╔═╡ 65398796-73ab-4d98-9851-3bb162ac8cbc
begin      # Create a model for the point spread function (PSF)
	psf_widths  = [0.5, 1.0, 2.0]
	psf_weights = [0.8, 0.15, 0.05]
	psf_model = GaussianMixtureConvolutionKernel(psf_widths,psf_weights)
end

# ╔═╡ 0aafec61-ff44-49e2-95e9-d3506ac6afa7
# Create a functor (function object) that computes a model for the the convolution of the raw spectrum with the PSF model
conv_spectrum = ConvolvedSpectrum(raw_spectrum,psf_model)

# ╔═╡ dbf05374-1d89-4f30-b4b4-6cf57631f8b7
begin
	plot(lambdas,raw_spectrum.(lambdas),xlabel="λ", ylabel="Flux", label="Raw spectrum", legend=:bottomright)
	plot!(lambdas,conv_spectrum.(lambdas), label="Convolved spectrum")
end

# ╔═╡ f2b23082-98bc-4be1-bb6d-cac8facb8a46
let
	plt = plot(view(lambdas,idx_plot),raw_spectrum.(view(lambdas,idx_plot)),xlabel="λ", ylabel="Flux", label="Raw spectrum", legend=:bottomright)
	plot!(plt,view(lambdas,idx_plot),conv_spectrum.(view(lambdas,idx_plot)), label="Convolved spectrum")
	ylims!(plt,0,1.01)
end

# ╔═╡ 6ccce964-0439-4707-adf9-e171fd703609
if true
	result_spec_vec_serial = conv_spectrum(lambdas)
	stats_spec_vec_serial = @timed conv_spectrum(lambdas)
	(;  time=stats_spec_vec_serial.time, bytes=stats_spec_vec_serial.bytes)
end

# ╔═╡ a172be44-1ac0-4bd8-a3d1-bac5666ab68e
if true
 	result_spec_serial_broadcast = conv_spectrum.(lambdas)
	stats_spec_serial_broadcast = @timed conv_spectrum.(lambdas)
	(;  time=stats_spec_serial_broadcast.time,
		bytes=stats_spec_serial_broadcast.bytes )
end

# ╔═╡ 215011e0-5977-43f8-bb65-83c09b3c07d8
if true
	result_spec_serial_map = map(conv_spectrum,lambdas)
	stats_spec_serial_map = @timed map(conv_spectrum,lambdas)
	(;  time=stats_spec_serial_map.time, bytes=stats_spec_serial_map.bytes )
end

# ╔═╡ c7121d63-b1ff-4c38-8579-e1adbfef48ef
if !ismissing(response_1b)
	result_spec_ThreadsXmap = ThreadsX.map(conv_spectrum,lambdas)
	stats_spec_ThreadsXmap = @timed ThreadsX.map(conv_spectrum,lambdas)
	(;  time=stats_spec_ThreadsXmap.time,
		bytes=stats_spec_ThreadsXmap.bytes )
end

# ╔═╡ 0e9664ec-98d8-49d4-a376-24d4770c4c8f
if !ismissing(response_1c)
	ThreadsX.map(conv_spectrum,lambdas,basesize=batchsize_for_ThreadsXmap)
	walltime_ThreadsXmap_batched = @elapsed ThreadsX.map(conv_spectrum,lambdas,basesize=batchsize_for_ThreadsXmap)
end

# ╔═╡ 4b9a98ba-1731-4707-89a3-db3b5ac3a79b
function calc_spectrum_loop(x::AbstractArray, spectrum::T) where T<:AbstractSpectrum
    out = zeros(length(x))
    for i in 1:length(x)
        @inbounds out[i] = spectrum(x[i])
    end
    return out
end

# ╔═╡ 9941061a-ad42-46b0-9d0f-7584ebca7c62
if true
	result_spec_serial_loop = calc_spectrum_loop(lambdas,conv_spectrum)
	stats_spec_serial_loop = @timed calc_spectrum_loop(lambdas,conv_spectrum)
	(;  time=stats_spec_serial_loop.time,
		bytes=stats_spec_serial_loop.bytes )
end

# ╔═╡ e55c802d-7923-458f-af42-d951e82e029b
function calc_spectrum_threaded_for_loop(x::AbstractArray, spectrum::T) where T<:AbstractSpectrum
    out = zeros(length(x))
    Threads.@threads for i in 1:length(x)
        @inbounds out[i] = spectrum(x[i])
    end
    return out
end

# ╔═╡ b3a6004f-9d10-4582-832a-8917b701f2ad
if !ismissing(response_1e)
	result_spec_threaded_loop = calc_spectrum_threaded_for_loop(lambdas,conv_spectrum)
	stats_spec_threaded_loop = @timed calc_spectrum_threaded_for_loop(lambdas,conv_spectrum)
	(;  time=stats_spec_threaded_loop.time,
		bytes=stats_spec_threaded_loop.bytes )
end

# ╔═╡ c65aa7b6-d85e-4efa-a2ee-1b615155796e
function calc_spectrum_threadsX_foreach(x::AbstractArray, spectrum::T ) where { T<:AbstractSpectrum }
    out = zeros(length(x))
	ThreadsX.foreach(eachindex(out, x)) do I
           @inbounds out[I] = spectrum(x[I])
    end
    return out
end

# ╔═╡ d1beea61-776f-4841-97e4-8d423ac22820
if true
	result_spec_threadsX_foreach = calc_spectrum_threadsX_foreach(lambdas,conv_spectrum)
	stats_spec_threadsX_foreach = @timed calc_spectrum_threadsX_foreach(lambdas,conv_spectrum)
	(;  time=stats_spec_threadsX_foreach.time,
		bytes=stats_spec_threadsX_foreach.bytes )
end

# ╔═╡ 9b734e9c-f571-4a09-9744-221dcd55b4bf
function calc_spectrum_flloop(x::AbstractArray, spectrum::T, ex::FLoops.Executor = ThreadedEx() ) where { T<:AbstractSpectrum }
    out = zeros(length(x))
     @floop ex for i in eachindex(out, x)
        @inbounds out[i] = spectrum(x[i])
    end
    return out
end

# ╔═╡ c2c68b93-1cd4-4a38-9dd9-47ce2d591907
if true
	result_spec_flloop = calc_spectrum_flloop(lambdas,conv_spectrum)
	stats_spec_flloop = @timed calc_spectrum_flloop(lambdas,conv_spectrum)
	(;  time=stats_spec_flloop.time, bytes=stats_spec_flloop.bytes )
end

# ╔═╡ 398ba928-899f-4843-ad58-25df67c81ffe
function calc_mse_broadcasted(lambdas::AbstractArray, spec1::AbstractSpectrum, spec2::AbstractSpectrum, v::Number)
	c = ModelSpectrum.speed_of_light
	z = v/c
	spec2_shifted = doppler_shifted_spectrum(spec2,z)
	mse = sum((spec1.(lambdas) .- spec2_shifted.(lambdas)).^2)
	mse /= length(lambdas)
end

# ╔═╡ 9f8667f3-4104-4642-b2d9-a6d12a6fa5d3
begin
	result_mse_broadcasted = calc_mse_broadcasted(lambdas,conv_spectrum,conv_spectrum,v)
	stats_mse_broadcasted = @timed calc_mse_broadcasted(lambdas,conv_spectrum,conv_spectrum,v)
end

# ╔═╡ 536fe0c4-567c-4bda-8c95-347f183c007b
function calc_mse_loop(lambdas::AbstractArray, spec1::AbstractSpectrum, spec2::AbstractSpectrum,  v::Number; ex = ThreadedEx())
	c = ModelSpectrum.speed_of_light
	z = v/c
	spec2_shifted = doppler_shifted_spectrum(spec2,z)
	tmp1 = spec1(first(lambdas))
    tmp2 = spec2_shifted(first(lambdas))
	mse = zero(promote_type(typeof(tmp1),typeof(tmp2)))
	for i in eachindex(lambdas)
        @inbounds l = lambdas[i]
		flux1 = spec1(l)
        flux2 = spec2_shifted(l)
		mse += (flux1-flux2)^2
    end
	mse /= length(lambdas)
    return mse
end

# ╔═╡ db96a6c9-8352-47f3-8319-9c373aa03ff4
if true
	result_mse_loop = calc_mse_loop(lambdas,conv_spectrum,conv_spectrum,v)
	stats_mse_loop = @timed calc_mse_loop(lambdas,conv_spectrum,conv_spectrum,v)
end

# ╔═╡ 6e52c719-e9fc-478a-9709-49e250a27d6b
md"""
As expected, the $(floor(Int,stats_mse_loop.bytes//1024^2)) MB allocated when we compute the mean squared error between *two* simulated spectra is very nearly twice the $(floor(Int,stats_spec_serial_loop.bytes//1024^2)) MB allocated by the serial for loop to compute the one spectrum at each wavelength.
"""

# ╔═╡ e36cda69-d300-4156-9bef-a372f94306d9
md"""
Similarly, it's likely that the wall time for the serial loop to compute the mean squared error $(round(stats_mse_loop.time,digits=3)) sec
is nearly twice that of the serial loop to compute one spectrum $(round(stats_spec_serial_loop.time,digits=3)) sec.
So far it doesn't seem particularly interesting.
"""

# ╔═╡ 1c1ccc51-e32a-4881-b892-095d2be55916
function calc_mse_flloop(lambdas::AbstractArray, spec1::AbstractSpectrum, spec2::AbstractSpectrum,  v::Number; ex = ThreadedEx())
	c = ModelSpectrum.speed_of_light
	z = v/c
	spec2_shifted = doppler_shifted_spectrum(spec2,z)
	tmp1 = spec1(first(lambdas))
    tmp2 = spec2_shifted(first(lambdas))
	mse = zero(promote_type(typeof(tmp1),typeof(tmp2)))
	@floop ex for i in eachindex(lambdas)
        @inbounds l = lambdas[i]
		flux1 = spec1(l)
        flux2 = spec2_shifted(l)
		@reduce(mse += (flux1-flux2)^2)
    end
	mse /= length(lambdas)
    return mse
end

# ╔═╡ b0e08212-7e12-4d54-846f-5b0863c37236
if !ismissing(response_1f)
	result_mse_flloop = calc_mse_flloop(lambdas,conv_spectrum,conv_spectrum,v)
	stats_mse_flloop = @timed calc_mse_flloop(lambdas,conv_spectrum,conv_spectrum,v)
	(;  time=stats_mse_flloop.time, bytes=stats_mse_flloop.bytes )
end

# ╔═╡ 3183c6ac-5acd-4770-a638-c4c6ba3f7c4f
if !ismissing(response_1f)
md"""
1g.  How did the performance of `calc_mse_flloop` compare to the performance of `calc_mse_loop`?  Was the wall time for the parallel loop to compute the mean squared error  $(stats_mse_flloop.time) sec
nearly twice that of the parallel loop to compute one spectrum $(stats_spec_flloop.time) sec?  Try to explain the main differences.
"""
end

# ╔═╡ 17659ddb-d4e0-4a4b-b34c-8ac52d5dad45
function calc_mse_mapreduce(lambdas::AbstractArray, spec1::AbstractSpectrum, spec2::AbstractSpectrum, v::Number)
	c = ModelSpectrum.speed_of_light
	z = v/c
	spec2_shifted = doppler_shifted_spectrum(spec2,z)
	mse = mapreduce(λ->(spec1.(λ) .- spec2_shifted.(λ)).^2, +, lambdas)
	mse /= length(lambdas)
end

# ╔═╡ 2ef9e7e0-c856-4ef3-a08f-89817fc5fd60
begin
	result_mse_mapreduce_serial = calc_mse_mapreduce(lambdas,conv_spectrum,conv_spectrum,v)
	stats_mse_mapreduce_serial = @timed calc_mse_mapreduce(lambdas, conv_spectrum,conv_spectrum,v)
	(;  time=stats_mse_mapreduce_serial.time, bytes=stats_mse_mapreduce_serial.bytes )
end

# ╔═╡ 1778899b-8f05-4b1f-acb5-32af1ace08ee
function calc_mse_mapreduce_threadsx(lambdas::AbstractArray, spec1::AbstractSpectrum, spec2::AbstractSpectrum,  v::Number; basesize::Integer = 1)
	c = ModelSpectrum.speed_of_light
	z = v/c
	spec2_shifted = doppler_shifted_spectrum(spec2,z)
	mse = ThreadsX.mapreduce(λ->(spec1.(λ) .- spec2_shifted.(λ)).^2, +, lambdas, basesize=basesize)
	mse /= length(lambdas)
end

# ╔═╡ 9e78bfc1-fb4e-4626-b387-c2f83bed6ef0
begin
	result_mse_mapreduce_threadsx = calc_mse_mapreduce_threadsx(lambdas,conv_spectrum,conv_spectrum,v)
	stats_mse_mapreduce_threadsx = @timed calc_mse_mapreduce_threadsx(lambdas,conv_spectrum,conv_spectrum,v)
	(;  time=stats_mse_mapreduce_threadsx.time, bytes=stats_mse_mapreduce_threadsx.bytes )

end

# ╔═╡ a661d895-d3d7-4e96-a08f-55b125ed1d40
begin
	result_mse_mapreduce_threadsx_batched = calc_mse_mapreduce_threadsx(lambdas,conv_spectrum,conv_spectrum,v; basesize=mapreduce_batchsize)
	stats_mse_mapreduce_threadsx_batched = @timed calc_mse_mapreduce_threadsx(lambdas,conv_spectrum,conv_spectrum,v; basesize=mapreduce_batchsize)
	(;  time=stats_mse_mapreduce_threadsx_batched.time, bytes=stats_mse_mapreduce_threadsx_batched.bytes )
end

# ╔═╡ 87df5b25-0d2f-4f81-80f1-aaf6c9f89ce3
# response_1i:
function calc_χ²_my_way(lambdas::AbstractArray, spec1::AbstractSpectrum, spec2::AbstractSpectrum, σ1::AbstractArray, σ2::AbstractArray, v::Number; #= any optional parameters? =# )
    # INSERT YOUR CODE HERE
    return missing
end

# ╔═╡ 4dec4888-08db-4965-b27a-fc44f316b529
begin
    if !@isdefined(calc_χ²_my_way)
		func_not_defined(:calc_χ²_my_way)
    elseif ismissing(calc_χ²_my_way(lambdas,conv_spectrum, conv_spectrum, σ_obs1, σ_obs2, 0.0))
    	still_missing()
	else
		md"I've provided some tests below to help you recognize if your parallelized version is working well."
	end
end

# ╔═╡ 6f411bcc-7084-43c3-a88b-b56ba77b5732
begin
    calc_χ²_my_way, lambdas, conv_spectrum, σ_obs1, σ_obs2
    @test abs(calc_χ²_my_way(lambdas,conv_spectrum, conv_spectrum, σ_obs1, σ_obs2, 0.0 )) < 1e-8
end

# ╔═╡ a9601654-8263-425e-8d8f-c5bbeacbbe06
begin
function calc_χ²_loop(lambdas::AbstractArray, spec1::AbstractSpectrum, spec2::AbstractSpectrum, σ1::AbstractArray, σ2::AbstractArray, v::Number )
    @assert size(lambdas) == size(σ1) == size(σ2)
    c = ModelSpectrum.speed_of_light
    z = v/c
    spec2_shifted = doppler_shifted_spectrum(spec2,z)
    tmp1 = spec1(first(lambdas))
    tmp2 = spec2_shifted(first(lambdas))
    χ² = zero(promote_type(typeof(tmp1),typeof(tmp2),eltype(σ1),eltype(σ2)))
    for i in eachindex(lambdas)
        @inbounds l = lambdas[i]
        flux1 = spec1(l)
        flux2 = spec2_shifted(l)
        @inbounds χ² += (flux1-flux2)^2/(σ1[i]^2+σ2[i]^2)
    end
    return χ²
end
	# for making urls to this cell
	linkto_calc_χ²_loop = "#" * (PlutoRunner.currently_running_cell_id[] |> string) 
end

# ╔═╡ 8737797c-6563-4513-a5fc-fde9681b4c63
Markdown.parse("""
1j.  Before parallelizing your project code for shared memory, it may be good to get some practice parallelizing a simple function very similar to what's already been done above.  Try parallelizing the function `calc_χ²` by writing a function `calc_χ²_my_way` in the cell below.   You can parallel the calculation of calculating χ² using any one of the parallelization strategies demonstrated above.  I'd suggest trying to use the one that you plan to use for your project.  Feel free to refer to the serial function [`calc_χ²` at bottom of notebook]($linkto_calc_χ²_loop).
""")

# ╔═╡ 3c5ee822-b938-4848-b2b0-f0de2e65b4db
begin
    calc_χ²_my_way, lambdas, conv_spectrum, σ_obs1, σ_obs2
    @test calc_χ²_my_way(lambdas,conv_spectrum, conv_spectrum, σ_obs1, σ_obs2, 10.0 ) ≈ calc_χ²_loop(lambdas,conv_spectrum, conv_spectrum, σ_obs1, σ_obs2, 10.0 )
end


# ╔═╡ Cell order:
# ╟─85aad005-eac0-4f71-a32c-c8361c31813b
# ╟─bdf61711-36e0-40d5-b0c5-3bac20a25aa3
# ╟─629442ba-a968-4e35-a7cb-d42a0a8783b4
# ╟─0bee1c3c-b130-49f2-baa4-efd8e3b49fdc
# ╟─f76f329a-8dde-4790-96f2-ade735643aeb
# ╟─0e4d7808-47e2-4740-ab93-5d3973eecaa8
# ╟─8a50e9fa-031c-4912-8a2d-466e6a9a9935
# ╟─7df5fc86-889f-4a5e-ac2b-8c6f68d7c32e
# ╟─571cab3f-771e-4464-959e-f351194049e2
# ╠═0c775b35-702e-4664-bd23-7557e4e189f4
# ╟─3059f3c2-cabf-4e20-adaa-9b6d0c07184f
# ╟─4fa907d0-c556-45df-8056-72041edcf430
# ╠═73e5e40a-1e59-41ed-a48d-7fb99f5a6755
# ╠═f97f1815-50a2-46a9-ac20-e4a3e34d898c
# ╟─53da8d7a-8620-4fe5-81ba-f615d2d4ed2a
# ╠═cc1418c8-3261-4c70-bc19-2921695570a6
# ╠═7f724449-e90e-4f8b-b13c-9640a498893c
# ╠═c85e51b2-2d3d-46a2-8f3f-03b289cab288
# ╟─907766c5-f084-4ddc-bb52-336cb037d521
# ╠═0bcde4df-1e31-4774-a31f-bd451bb6f758
# ╟─c41d65e3-ea35-4f97-90a1-bfeaeaf927ad
# ╟─6e617a7c-a640-4cb3-9451-28a0036d8fdc
# ╟─5e6c430a-cd2f-4169-a5c7-a92acef813ac
# ╟─c31cf36c-21ec-46f1-96aa-b014ff094f8a
# ╠═4effbde2-2764-4c51-a9d0-a2db82f60862
# ╟─7026e51d-c3e4-4503-9f35-71074b0c2f1a
# ╠═ad302f2b-69dc-4559-ba12-d7fb2e8e689e
# ╠═86b8dd31-1261-4fb9-bfd3-13f6f01e7790
# ╟─16ad0225-c7d6-455b-8eb0-3e93c9f9f91a
# ╠═65398796-73ab-4d98-9851-3bb162ac8cbc
# ╠═0aafec61-ff44-49e2-95e9-d3506ac6afa7
# ╟─324a9a25-1ec4-4dc2-a7ca-e0f1f56dbf66
# ╠═52127f57-9a07-451a-bb24-c1f3c5581f0a
# ╟─dbf05374-1d89-4f30-b4b4-6cf57631f8b7
# ╟─75948469-1347-45e2-9281-f366b41d0e04
# ╟─f2b23082-98bc-4be1-bb6d-cac8facb8a46
# ╟─4d1cf57f-b394-4f37-98c3-0d765f4ee635
# ╟─cddd761a-f051-4338-9e40-d35e050060d3
# ╟─ee96411d-e3fa-442b-b0fe-10d6ede37b6a
# ╟─b92aad2e-8a3b-4edf-ae7e-6e3cff6eead4
# ╟─e5f9fa06-9fbb-40a8-92de-71523775d257
# ╟─b195ebd2-9584-40b8-ae3e-6d9ce88b5398
# ╠═658f73c3-1e7a-47da-9130-06673f484ba1
# ╠═1c069610-4468-4d10-98f7-99662c26bdda
# ╟─d6d3a2d1-241e-44c1-a11b-5bfb2b3c5f4b
# ╟─0344a74d-456b-44f0-84dc-c2fdbd41a379
# ╠═6ccce964-0439-4707-adf9-e171fd703609
# ╠═a172be44-1ac0-4bd8-a3d1-bac5666ab68e
# ╟─51adffd7-8fb6-4ed2-8510-303a37d6efc3
# ╟─71d943e3-761a-4337-b412-b0b768483bc2
# ╟─db1583f4-61cb-43e0-9326-d6c15d8fad5a
# ╠═ca9c7d9e-e6cc-46cc-8a9b-ccda123591a2
# ╠═215011e0-5977-43f8-bb65-83c09b3c07d8
# ╟─f108d26b-6c75-4eb6-9e88-a60ec038a73c
# ╟─e71cede9-382e-47e2-953a-2fa96ed50002
# ╟─4d54b6a7-3fc0-4c63-8a9d-d683aa4ecefe
# ╟─21f305db-24e1-47d1-b1f4-be04ca91780e
# ╟─a44a3478-541d-40d6-9d99-04b918c16bfb
# ╠═4b9a98ba-1731-4707-89a3-db3b5ac3a79b
# ╠═9941061a-ad42-46b0-9d0f-7584ebca7c62
# ╟─96914ff8-56c8-4cc8-96bc-fd3d13f7e4ce
# ╟─32685a28-54d9-4c0d-8940-e82843d2cab2
# ╟─3717d201-0bc3-4e3c-8ecd-d835e58f6821
# ╟─496e8c5e-251b-4448-8c59-541877d752c1
# ╟─04bcafcd-1d2f-4ce5-893f-7ec5bb05f9ed
# ╠═ca8ceb27-86ea-4b90-a1ae-86d794c9fc98
# ╟─4ad081a2-b5c2-48ff-9a28-ec9c8d9f0d0e
# ╠═c7121d63-b1ff-4c38-8579-e1adbfef48ef
# ╟─2399ce76-b6da-4a61-bcda-aee22dd275f8
# ╠═a25c6705-54f4-4bad-966e-a8f13ae4c711
# ╟─739136b1-6b01-44c0-bbfd-dcb490d1e191
# ╟─dcce9a84-a9b1-47c1-8e08-7575cb299b56
# ╠═fb063bc5-22bc-4b32-8fcb-5fbc4765c8b5
# ╠═0e9664ec-98d8-49d4-a376-24d4770c4c8f
# ╟─90c9d079-4bbc-4609-aa12-afa41a74b2fb
# ╟─0edbb2db-4db8-4dc4-9a73-f7ff86e6f577
# ╟─a944fdea-f41b-4a5f-95ac-e5f4074d4290
# ╟─bd81357b-c461-458e-801c-610893dd5ea1
# ╟─0e0c25d4-35b9-429b-8223-90e9e8be90f9
# ╠═e55c802d-7923-458f-af42-d951e82e029b
# ╟─5a63ebd6-3e18-49ee-8d1d-4bb2da6419b6
# ╠═86e7d984-c128-4d2e-8599-3bc70db87a1d
# ╟─c69c0a4a-b90b-414c-883d-3aa50c04b5e1
# ╠═b3a6004f-9d10-4582-832a-8917b701f2ad
# ╟─791041e9-d277-4cac-a5ac-1d6ec52e0287
# ╠═c65aa7b6-d85e-4efa-a2ee-1b615155796e
# ╠═d1beea61-776f-4841-97e4-8d423ac22820
# ╠═9b734e9c-f571-4a09-9744-221dcd55b4bf
# ╠═c2c68b93-1cd4-4a38-9dd9-47ce2d591907
# ╟─7c367a0b-c5b9-459b-9ccf-e07c84e0b32a
# ╟─2b00f6fc-9bfd-48d6-a4d8-ac95f7e71faa
# ╟─ea002e89-9f4e-441e-8998-5e9c99bb27e0
# ╟─d43525da-e0a2-4d2f-9dbb-bf187eebf6c1
# ╟─547ad5ba-06ad-4707-a7ef-e444cf88ae53
# ╟─7ba35a63-ac61-434b-b759-95d505f62d9e
# ╠═398ba928-899f-4843-ad58-25df67c81ffe
# ╠═cee9c93d-cf7b-4da1-b4bb-b544b7cc104c
# ╠═9f8667f3-4104-4642-b2d9-a6d12a6fa5d3
# ╟─3ac01c04-52e3-497e-8c29-8c704e23ae39
# ╟─790377a7-1301-44a8-b300-418567737373
# ╠═536fe0c4-567c-4bda-8c95-347f183c007b
# ╠═db96a6c9-8352-47f3-8319-9c373aa03ff4
# ╟─6e52c719-e9fc-478a-9709-49e250a27d6b
# ╟─e36cda69-d300-4156-9bef-a372f94306d9
# ╟─161ea6af-5661-44e1-ae40-1b581b636c25
# ╠═1c1ccc51-e32a-4881-b892-095d2be55916
# ╟─7def3535-6f90-4bf8-b86f-aac278666663
# ╠═1989da2a-1fe2-49a0-b279-5925ae4b428c
# ╟─8d7c27d5-4a07-4ab4-9ece-94fdb7053f73
# ╠═b0e08212-7e12-4d54-846f-5b0863c37236
# ╟─3183c6ac-5acd-4770-a638-c4c6ba3f7c4f
# ╠═8e9b1e02-2bc0-49d2-b7ed-38de877ebe77
# ╟─ba62f716-b1b5-4d11-91f2-ed121b48216c
# ╟─bbdd495c-f2c6-4264-a4e9-5083753eb410
# ╟─383aa611-e115-482e-873c-4487e53d457f
# ╟─2c6fa743-3dec-417b-b05a-17bb52b5d39d
# ╠═17659ddb-d4e0-4a4b-b34c-8ac52d5dad45
# ╠═2ef9e7e0-c856-4ef3-a08f-89817fc5fd60
# ╟─ae47ef38-e8d0-40b9-9e61-3ab3ca7e7a49
# ╟─aad94861-e2b3-417d-b640-b821e53adb23
# ╠═1778899b-8f05-4b1f-acb5-32af1ace08ee
# ╠═9e78bfc1-fb4e-4626-b387-c2f83bed6ef0
# ╟─f1c0321b-7811-42b1-9d0c-9c69f43d7e1a
# ╠═df044a68-605f-4347-832a-68090ee07950
# ╠═a661d895-d3d7-4e96-a08f-55b125ed1d40
# ╟─3f01d534-b01d-4ab4-b3cd-e809b02563a9
# ╠═d16adf94-72c3-480d-bd92-738e806068f8
# ╟─56c5b496-a063-459a-8686-22fc70b6a214
# ╟─c4ff4add-ab3c-4585-900e-41f17e905ac5
# ╠═ac18f1ca-0f60-4436-9d8a-797b3dfd8657
# ╟─e8082779-143d-4562-81f3-d493679cf3c7
# ╟─8737797c-6563-4513-a5fc-fde9681b4c63
# ╠═87df5b25-0d2f-4f81-80f1-aaf6c9f89ce3
# ╟─4dec4888-08db-4965-b27a-fc44f316b529
# ╠═bd77bc71-ffdf-4ba1-b1ee-6f2a69044e6f
# ╠═6f411bcc-7084-43c3-a88b-b56ba77b5732
# ╠═3c5ee822-b938-4848-b2b0-f0de2e65b4db
# ╟─3b50062c-99c1-4f68-aabe-2d40d4ad7504
# ╟─d83a282e-cb2b-4837-bfd4-8404b3722e3a
# ╟─c9cf6fb3-0146-42e6-aaae-24e97254c805
# ╠═76730d06-06da-4466-8814-2096b221090f
# ╠═73358bcf-4129-46be-bef4-f623b11e245b
# ╠═a9601654-8263-425e-8d8f-c5bbeacbbe06
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
