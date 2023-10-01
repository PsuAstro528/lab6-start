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

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
CpuId = "adafc99b-e345-5852-983c-f28acb93d879"
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
FLoops = "cc61a311-1640-44b5-9fba-1b764f453329"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoTeachingTools = "661c6b06-c737-4d37-b85c-46df65de6f69"
PlutoTest = "cb4044da-4d16-4ffa-a6a3-8cad7f73ebdc"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
QuadGK = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
ThreadsX = "ac1d9e8a-700a-412c-b207-f0111f4b6c0d"

[compat]
BenchmarkTools = "~1.3.2"
CpuId = "~0.3.1"
Distributions = "~0.25.102"
FLoops = "~0.2.1"
Plots = "~1.39.0"
PlutoTeachingTools = "~0.2.13"
PlutoTest = "~0.2.2"
PlutoUI = "~0.7.52"
QuadGK = "~2.9.1"
StaticArrays = "~1.6.5"
ThreadsX = "~0.1.11"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.9.2"
manifest_format = "2.0"
project_hash = "8328a39489a3d6219537e04dba37470ba6f57212"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "91bd53c39b9cbfb5ef4b015e8b582d344532bd0a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.2.0"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "76289dc51920fdc6e0013c872ba9551d54961c24"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.6.2"
weakdeps = ["StaticArrays"]

    [deps.Adapt.extensions]
    AdaptStaticArraysExt = "StaticArrays"

[[deps.ArgCheck]]
git-tree-sha1 = "a3a402a35a2f7e0b87828ccabbd5ebfbebe356b4"
uuid = "dce04be8-c92d-5529-be00-80e4d2c0e197"
version = "2.3.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.BangBang]]
deps = ["Compat", "ConstructionBase", "InitialValues", "LinearAlgebra", "Requires", "Setfield", "Tables"]
git-tree-sha1 = "e28912ce94077686443433c2800104b061a827ed"
uuid = "198e06fe-97b7-11e9-32a5-e1d131e6ad66"
version = "0.3.39"

    [deps.BangBang.extensions]
    BangBangChainRulesCoreExt = "ChainRulesCore"
    BangBangDataFramesExt = "DataFrames"
    BangBangStaticArraysExt = "StaticArrays"
    BangBangStructArraysExt = "StructArrays"
    BangBangTypedTablesExt = "TypedTables"

    [deps.BangBang.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    TypedTables = "9d95f2ec-7b3d-5a63-8d20-e2491e220bb9"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.Baselet]]
git-tree-sha1 = "aebf55e6d7795e02ca500a689d326ac979aaf89e"
uuid = "9718e550-a3fa-408a-8086-8db961cd8217"
version = "0.1.1"

[[deps.BenchmarkTools]]
deps = ["JSON", "Logging", "Printf", "Profile", "Statistics", "UUIDs"]
git-tree-sha1 = "d9a9701b899b30332bbcb3e1679c41cce81fb0e8"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.3.2"

[[deps.BitFlags]]
git-tree-sha1 = "43b1a4a8f797c1cddadf60499a8a077d4af2cd2d"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.7"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "4b859a208b2397a7a623a03449e4636bdb17bcf2"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+1"

[[deps.Calculus]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f641eb0a4f00c343bbc32346e1217b86f3ce9dad"
uuid = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
version = "0.5.1"

[[deps.CodeTracking]]
deps = ["InteractiveUtils", "UUIDs"]
git-tree-sha1 = "a1296f0fe01a4c3f9bf0dc2934efbf4416f5db31"
uuid = "da1fd8a2-8d9e-5ec2-8556-3022fb5608a2"
version = "1.3.4"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "02aa26a4cf76381be7f66e020a3eddeb27b0a092"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.2"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "67c1f244b991cad9b0aa4b7540fb758c2488b129"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.24.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "a1f44953f2382ebb937d60dafbe2deea4bd23249"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.10.0"
weakdeps = ["SpecialFunctions"]

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "fc08e5930ee9a4e03f84bfb5211cb54e7769758a"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.10"

[[deps.Compat]]
deps = ["UUIDs"]
git-tree-sha1 = "8a62af3e248a8c4bad6b32cbbe663ae02275e32c"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.10.0"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.0.5+0"

[[deps.CompositionsBase]]
git-tree-sha1 = "802bb88cd69dfd1509f6670416bd4434015693ad"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.2"

    [deps.CompositionsBase.extensions]
    CompositionsBaseInverseFunctionsExt = "InverseFunctions"

    [deps.CompositionsBase.weakdeps]
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.ConcurrentUtilities]]
deps = ["Serialization", "Sockets"]
git-tree-sha1 = "5372dbbf8f0bdb8c700db5367132925c0771ef7e"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.2.1"

[[deps.ConstructionBase]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "c53fc348ca4d40d7b371e71fd52251839080cbc9"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.5.4"

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseStaticArraysExt = "StaticArrays"

    [deps.ConstructionBase.weakdeps]
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.ContextVariablesX]]
deps = ["Compat", "Logging", "UUIDs"]
git-tree-sha1 = "25cc3803f1030ab855e383129dcd3dc294e322cc"
uuid = "6add18c4-b38d-439d-96f6-d6bc489c04c5"
version = "0.1.3"

[[deps.Contour]]
git-tree-sha1 = "d05d9e7b7aedff4e5b51a029dced05cfb6125781"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.2"

[[deps.CpuId]]
deps = ["Markdown"]
git-tree-sha1 = "fcbb72b032692610bfbdb15018ac16a36cf2e406"
uuid = "adafc99b-e345-5852-983c-f28acb93d879"
version = "0.3.1"

[[deps.DataAPI]]
git-tree-sha1 = "8da84edb865b0b5b0100c0666a9bc9a0b71c553c"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.15.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "3dbd312d370723b6bb43ba9d02fc36abade4518d"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.15"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DefineSingletons]]
git-tree-sha1 = "0fba8b706d0178b4dc7fd44a96a92382c9065c2c"
uuid = "244e2a9f-e319-4986-a169-4d1fe445cd52"
version = "0.1.2"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.Distributions]]
deps = ["FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns", "Test"]
git-tree-sha1 = "3d5873f811f582873bb9871fc9c451784d5dc8c7"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.102"

    [deps.Distributions.extensions]
    DistributionsChainRulesCoreExt = "ChainRulesCore"
    DistributionsDensityInterfaceExt = "DensityInterface"

    [deps.Distributions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DensityInterface = "b429d917-457f-4dbc-8f4c-0cc954292b1d"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.DualNumbers]]
deps = ["Calculus", "NaNMath", "SpecialFunctions"]
git-tree-sha1 = "5837a837389fccf076445fce071c8ddaea35a566"
uuid = "fa6b7ba4-c1ee-5f82-b5fc-ecf0adba8f74"
version = "0.6.8"

[[deps.EpollShim_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8e9441ee83492030ace98f9789a654a6d0b1f643"
uuid = "2702e6a9-849d-5ed8-8c21-79e8b8f9ee43"
version = "0.0.20230411+0"

[[deps.ExceptionUnwrapping]]
deps = ["Test"]
git-tree-sha1 = "e90caa41f5a86296e014e148ee061bd6c3edec96"
uuid = "460bff9d-24e4-43bc-9d9f-a8973cb893f4"
version = "0.1.9"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "4558ab818dcceaab612d1bb8c19cee87eda2b83c"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.5.0+0"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Pkg", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "74faea50c1d007c85837327f6775bea60b5492dd"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.2+2"

[[deps.FLoops]]
deps = ["BangBang", "Compat", "FLoopsBase", "InitialValues", "JuliaVariables", "MLStyle", "Serialization", "Setfield", "Transducers"]
git-tree-sha1 = "ffb97765602e3cbe59a0589d237bf07f245a8576"
uuid = "cc61a311-1640-44b5-9fba-1b764f453329"
version = "0.2.1"

[[deps.FLoopsBase]]
deps = ["ContextVariablesX"]
git-tree-sha1 = "656f7a6859be8673bf1f35da5670246b923964f7"
uuid = "b9860ae5-e623-471e-878b-f6a53c775ea6"
version = "0.1.1"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FillArrays]]
deps = ["LinearAlgebra", "Random"]
git-tree-sha1 = "a20eaa3ad64254c61eeb5f230d9306e937405434"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.6.1"
weakdeps = ["SparseArrays", "Statistics"]

    [deps.FillArrays.extensions]
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStatisticsExt = "Statistics"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[deps.Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "d8db6a5a2fe1381c1ea4ef2cab7c69c2de7f9ea0"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.13.1+0"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "d972031d28c8c8d9d7b41a536ad7bb0c2579caca"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.8+0"

[[deps.GR]]
deps = ["Artifacts", "Base64", "DelimitedFiles", "Downloads", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Preferences", "Printf", "Random", "Serialization", "Sockets", "TOML", "Tar", "Test", "UUIDs", "p7zip_jll"]
git-tree-sha1 = "27442171f28c952804dede8ff72828a96f2bfc1f"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.72.10"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "FreeType2_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Qt6Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "025d171a2847f616becc0f84c8dc62fe18f0f6dd"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.72.10+0"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "e94c92c7bf4819685eb80186d51c43e71d4afa17"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.76.5+0"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HTTP]]
deps = ["Base64", "CodecZlib", "ConcurrentUtilities", "Dates", "ExceptionUnwrapping", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "5eab648309e2e060198b45820af1a37182de3cce"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.10.0"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[deps.HypergeometricFunctions]]
deps = ["DualNumbers", "LinearAlgebra", "OpenLibm_jll", "SpecialFunctions"]
git-tree-sha1 = "f218fe3736ddf977e0e772bc9a586b2383da2685"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.23"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "c47c5fa4c5308f27ccaac35504858d8914e102f9"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.4"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "d75853a0bdbfb1ac815478bacd89cd27b550ace6"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.3"

[[deps.InitialValues]]
git-tree-sha1 = "4da0f88e9a39111c2fa3add390ab15f3a44f3ca3"
uuid = "22cec73e-a1b8-11e9-2c92-598750a2cf9c"
version = "0.3.1"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLFzf]]
deps = ["Pipe", "REPL", "Random", "fzf_jll"]
git-tree-sha1 = "f377670cda23b6b7c1c0b3893e37451c5c1a2185"
uuid = "1019f520-868f-41f5-a6de-eb00f4b6a39c"
version = "0.1.5"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "7e5d6779a1e09a36db2a7b6cff50942a0a7d0fca"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.5.0"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6f2675ef130a300a112286de91973805fcc5ffbc"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.91+0"

[[deps.JuliaInterpreter]]
deps = ["CodeTracking", "InteractiveUtils", "Random", "UUIDs"]
git-tree-sha1 = "81dc6aefcbe7421bd62cb6ca0e700779330acff8"
uuid = "aa1ae85d-cabe-5617-a682-6adf51b2e16a"
version = "0.9.25"

[[deps.JuliaVariables]]
deps = ["MLStyle", "NameResolution"]
git-tree-sha1 = "49fb3cb53362ddadb4415e9b73926d6b40709e70"
uuid = "b14d175d-62b4-44ba-8fb7-3064adc8c3ec"
version = "0.2.4"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bf36f528eec6634efc60d7ec062008f171071434"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "3.0.0+1"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f689897ccbe049adb19a065c495e75f372ecd42b"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "15.0.4+0"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "f2355693d6778a178ade15952b7ac47a4ff97996"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.0"

[[deps.Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Printf", "Requires"]
git-tree-sha1 = "f428ae552340899a935973270b8d98e5a31c49fe"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.16.1"

    [deps.Latexify.extensions]
    DataFramesExt = "DataFrames"
    SymEngineExt = "SymEngine"

    [deps.Latexify.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.3"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "7.84.0+0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.10.2+0"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[deps.Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "6f73d1dd803986947b2c750138528a999a6c7733"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.6.0+0"

[[deps.Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "f9557a255370125b405568f9767d6d195822a175"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.17.0+0"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "2da088d113af58221c52828a80378e16be7d037a"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.5.1+1"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "7d6dd4e9212aebaeed356de34ccf262a3cd415aa"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.26"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "0d097476b6c381ab7906460ef1ef1638fbce1d91"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.0.2"

[[deps.LoweredCodeUtils]]
deps = ["JuliaInterpreter"]
git-tree-sha1 = "60168780555f3e663c536500aa790b6368adc02a"
uuid = "6f1432cf-f94c-5a45-995e-cdbf5db27b0b"
version = "2.3.0"

[[deps.MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[deps.MLStyle]]
git-tree-sha1 = "bc38dff0548128765760c79eb7388a4b37fae2c8"
uuid = "d8e11817-5142-5d16-987a-aa16d5891078"
version = "0.4.17"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "9ee1618cbf5240e6d4e0371d6f24065083f60c48"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.11"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "Random", "Sockets"]
git-tree-sha1 = "03a9b9718f5682ecb107ac9f7308991db4ce395b"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.7"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.2+0"

[[deps.Measures]]
git-tree-sha1 = "c13304c81eec1ed3af7fc20e75fb6b26092a1102"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.2"

[[deps.MicroCollections]]
deps = ["BangBang", "InitialValues", "Setfield"]
git-tree-sha1 = "629afd7d10dbc6935ec59b32daeb33bc4460a42e"
uuid = "128add7d-3638-4c79-886c-908ea0c25c34"
version = "0.1.4"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "f66bdc5de519e8f8ae43bdc598782d35a25b1272"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.1.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.10.11"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "0877504529a3e5c3343c6f8b4c0381e57e4387e4"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.2"

[[deps.NameResolution]]
deps = ["PrettyPrint"]
git-tree-sha1 = "1a0fa0e9613f46c9b8c11eee38ebb4f590013c5e"
uuid = "71a1bf82-56d0-4bbc-8a3c-48b961074391"
version = "0.1.5"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.21+4"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+0"

[[deps.OpenSSL]]
deps = ["BitFlags", "Dates", "MozillaCACerts_jll", "OpenSSL_jll", "Sockets"]
git-tree-sha1 = "51901a49222b09e3743c65b8847687ae5fc78eb2"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.4.1"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a12e56c72edee3ce6b96667745e6cbbe5498f200"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.23+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "2e73fe17cac3c62ad1aebe70d44c963c3cfdc3e3"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.6.2"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.42.0+0"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "bf6085e8bd7735e68c210c6e5d81f9a6fe192060"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.19"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "716e24b21538abc91f6205fd1d8363f39b442851"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.7.2"

[[deps.Pipe]]
git-tree-sha1 = "6842804e7867b115ca9de748a0cf6b364523c16d"
uuid = "b98c9c47-44ae-5843-9183-064241ee97a0"
version = "1.3.0"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "64779bc4c9784fee475689a1752ef4d5747c5e87"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.42.2+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.9.2"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "1f03a2d339f42dca4a4da149c7e15e9b896ad899"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.1.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "f92e1315dadf8c46561fb9396e525f7200cdc227"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.3.5"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "JLFzf", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "PrecompileTools", "Preferences", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "RelocatableFolders", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun", "UnitfulLatexify", "Unzip"]
git-tree-sha1 = "ccee59c6e48e6f2edf8a5b64dc817b6729f99eb5"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.39.0"

    [deps.Plots.extensions]
    FileIOExt = "FileIO"
    GeometryBasicsExt = "GeometryBasics"
    IJuliaExt = "IJulia"
    ImageInTerminalExt = "ImageInTerminal"
    UnitfulExt = "Unitful"

    [deps.Plots.weakdeps]
    FileIO = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
    GeometryBasics = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"
    ImageInTerminal = "d8c32880-2388-543b-8c61-d9f865259254"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.PlutoHooks]]
deps = ["InteractiveUtils", "Markdown", "UUIDs"]
git-tree-sha1 = "072cdf20c9b0507fdd977d7d246d90030609674b"
uuid = "0ff47ea0-7a50-410d-8455-4348d5de0774"
version = "0.0.5"

[[deps.PlutoLinks]]
deps = ["FileWatching", "InteractiveUtils", "Markdown", "PlutoHooks", "Revise", "UUIDs"]
git-tree-sha1 = "8f5fa7056e6dcfb23ac5211de38e6c03f6367794"
uuid = "0ff47ea0-7a50-410d-8455-4348d5de0420"
version = "0.1.6"

[[deps.PlutoTeachingTools]]
deps = ["Downloads", "HypertextLiteral", "LaTeXStrings", "Latexify", "Markdown", "PlutoLinks", "PlutoUI", "Random"]
git-tree-sha1 = "542de5acb35585afcf202a6d3361b430bc1c3fbd"
uuid = "661c6b06-c737-4d37-b85c-46df65de6f69"
version = "0.2.13"

[[deps.PlutoTest]]
deps = ["HypertextLiteral", "InteractiveUtils", "Markdown", "Test"]
git-tree-sha1 = "17aa9b81106e661cffa1c4c36c17ee1c50a86eda"
uuid = "cb4044da-4d16-4ffa-a6a3-8cad7f73ebdc"
version = "0.2.2"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "e47cd150dbe0443c3a3651bc5b9cbd5576ab75b7"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.52"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "03b4c25b43cb84cee5c90aa9b5ea0a78fd848d2f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.0"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "00805cd429dcb4870060ff49ef443486c262e38e"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.1"

[[deps.PrettyPrint]]
git-tree-sha1 = "632eb4abab3449ab30c5e1afaa874f0b98b586e4"
uuid = "8162dcfd-2161-5ef2-ae6c-7681170c5f98"
version = "0.2.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.Profile]]
deps = ["Printf"]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"

[[deps.Qt6Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Vulkan_Loader_jll", "Xorg_libSM_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_cursor_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "libinput_jll", "xkbcommon_jll"]
git-tree-sha1 = "7c29f0e8c575428bd84dc3c72ece5178caa67336"
uuid = "c0090381-4147-56d7-9ebc-da0b1113ec56"
version = "6.5.2+2"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "9ebcd48c498668c7fa0e97a9cae873fbee7bfee1"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.9.1"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "PrecompileTools", "RecipesBase"]
git-tree-sha1 = "45cf9fd0ca5839d06ef333c8201714e888486342"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.6.12"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.Referenceables]]
deps = ["Adapt"]
git-tree-sha1 = "e681d3bfa49cd46c3c161505caddf20f0e62aaa9"
uuid = "42d2dcc6-99eb-4e98-b66c-637b7d73030e"
version = "0.1.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "90bc7a7c96410424509e4263e277e43250c05691"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.0"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.Revise]]
deps = ["CodeTracking", "Distributed", "FileWatching", "JuliaInterpreter", "LibGit2", "LoweredCodeUtils", "OrderedCollections", "Pkg", "REPL", "Requires", "UUIDs", "Unicode"]
git-tree-sha1 = "609c26951d80551620241c3d7090c71a73da75ab"
uuid = "295af30f-e4ad-537b-8983-00126c2a3abe"
version = "3.5.6"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "f65dcb5fa46aee0cf9ed6274ccbd597adc49aa7b"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.1"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6ed52fdd3382cf21947b15e8870ac0ddbff736da"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.4.0+0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "30449ee12237627992a99d5e30ae63e4d78cd24a"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.2.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "StaticArraysCore"]
git-tree-sha1 = "e2cc6d8c88613c05e1defb55170bf5ff211fbeac"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "1.1.1"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SimpleBufferStream]]
git-tree-sha1 = "874e8867b33a00e784c8a7e4b60afe9e037b74e1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.1.0"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "c60ec5c62180f27efea3ba2908480f8055e17cee"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.1.1"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "e2cfc4012a19088254b3950b85c3c1d8882d864d"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.3.1"

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

    [deps.SpecialFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"

[[deps.SplittablesBase]]
deps = ["Setfield", "Test"]
git-tree-sha1 = "e08a62abc517eb79667d0a29dc08a3b589516bb5"
uuid = "171d559e-b47b-412a-8079-5efa626c420e"
version = "0.1.15"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "StaticArraysCore"]
git-tree-sha1 = "0adf069a2a490c47273727e029371b31d44b72b2"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.6.5"
weakdeps = ["Statistics"]

    [deps.StaticArrays.extensions]
    StaticArraysStatisticsExt = "Statistics"

[[deps.StaticArraysCore]]
git-tree-sha1 = "36b3d696ce6366023a0ea192b4cd442268995a0d"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.2"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.9.0"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1ff449ad350c9c4cbc756624d6f8a8c3ef56d3ed"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.7.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "1d77abd07f617c4868c33d4f5b9e1dbb2643c9cf"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.2"

[[deps.StatsFuns]]
deps = ["HypergeometricFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "f625d686d5a88bcd2b15cd81f18f98186fdc0c9a"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.3.0"

    [deps.StatsFuns.extensions]
    StatsFunsChainRulesCoreExt = "ChainRulesCore"
    StatsFunsInverseFunctionsExt = "InverseFunctions"

    [deps.StatsFuns.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "Pkg", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "5.10.1+6"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "a1f34829d5ac0ef499f6d84428bd6b4c71f02ead"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.11.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.ThreadsX]]
deps = ["ArgCheck", "BangBang", "ConstructionBase", "InitialValues", "MicroCollections", "Referenceables", "Setfield", "SplittablesBase", "Transducers"]
git-tree-sha1 = "34e6bcf36b9ed5d56489600cf9f3c16843fa2aa2"
uuid = "ac1d9e8a-700a-412c-b207-f0111f4b6c0d"
version = "0.1.11"

[[deps.TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "9a6ae7ed916312b41236fcef7e0af564ef934769"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.13"

[[deps.Transducers]]
deps = ["Adapt", "ArgCheck", "BangBang", "Baselet", "CompositionsBase", "ConstructionBase", "DefineSingletons", "Distributed", "InitialValues", "Logging", "Markdown", "MicroCollections", "Requires", "Setfield", "SplittablesBase", "Tables"]
git-tree-sha1 = "53bd5978b182fa7c57577bdb452c35e5b4fb73a5"
uuid = "28d57a85-8fef-5791-bfe6-a80928e7c999"
version = "0.4.78"

    [deps.Transducers.extensions]
    TransducersBlockArraysExt = "BlockArrays"
    TransducersDataFramesExt = "DataFrames"
    TransducersLazyArraysExt = "LazyArrays"
    TransducersOnlineStatsBaseExt = "OnlineStatsBase"
    TransducersReferenceablesExt = "Referenceables"

    [deps.Transducers.weakdeps]
    BlockArrays = "8e7c35d0-a365-5155-bbbb-fb81a777f24e"
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    LazyArrays = "5078a376-72f3-5289-bfd5-ec5146d43c02"
    OnlineStatsBase = "925886fa-5bf2-5e8e-b522-a9147a512338"
    Referenceables = "42d2dcc6-99eb-4e98-b66c-637b7d73030e"

[[deps.Tricks]]
git-tree-sha1 = "aadb748be58b492045b4f56166b5188aa63ce549"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.7"

[[deps.URIs]]
git-tree-sha1 = "b7a5e99f24892b6824a954199a45e9ffcc1c70f0"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.5.0"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unitful]]
deps = ["Dates", "LinearAlgebra", "Random"]
git-tree-sha1 = "a72d22c7e13fe2de562feda8645aa134712a87ee"
uuid = "1986cc42-f94f-5a68-af5c-568840ba703d"
version = "1.17.0"

    [deps.Unitful.extensions]
    ConstructionBaseUnitfulExt = "ConstructionBase"
    InverseFunctionsUnitfulExt = "InverseFunctions"

    [deps.Unitful.weakdeps]
    ConstructionBase = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.UnitfulLatexify]]
deps = ["LaTeXStrings", "Latexify", "Unitful"]
git-tree-sha1 = "e2d817cc500e960fdbafcf988ac8436ba3208bfd"
uuid = "45397f5d-5981-4c77-b2b3-fc36d6e9b728"
version = "1.6.3"

[[deps.Unzip]]
git-tree-sha1 = "ca0969166a028236229f63514992fc073799bb78"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.2.0"

[[deps.Vulkan_Loader_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Wayland_jll", "Xorg_libX11_jll", "Xorg_libXrandr_jll", "xkbcommon_jll"]
git-tree-sha1 = "2f0486047a07670caad3a81a075d2e518acc5c59"
uuid = "a44049a8-05dd-5a78-86c9-5fde0876e88c"
version = "1.3.243+0"

[[deps.Wayland_jll]]
deps = ["Artifacts", "EpollShim_jll", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "7558e29847e99bc3f04d6569e82d0f5c54460703"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.21.0+1"

[[deps.Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4528479aa01ee1b3b4cd0e6faef0e04cf16466da"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.25.0+0"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Zlib_jll"]
git-tree-sha1 = "24b81b59bd35b3c42ab84fa589086e19be919916"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.11.5+0"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[deps.XZ_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "cf2c7de82431ca6f39250d2fc4aacd0daa1675c0"
uuid = "ffd25f8a-64ca-5728-b0f7-c24cf3aae800"
version = "5.4.4+0"

[[deps.Xorg_libICE_jll]]
deps = ["Libdl", "Pkg"]
git-tree-sha1 = "e5becd4411063bdcac16be8b66fc2f9f6f1e8fe5"
uuid = "f67eecfb-183a-506d-b269-f58e52b52d7c"
version = "1.0.10+1"

[[deps.Xorg_libSM_jll]]
deps = ["Libdl", "Pkg", "Xorg_libICE_jll"]
git-tree-sha1 = "4a9d9e4c180e1e8119b5ffc224a7b59d3a7f7e18"
uuid = "c834827a-8449-5923-a945-d239c165b7dd"
version = "1.2.3+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "afead5aba5aa507ad5a3bf01f58f82c8d1403495"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.6+0"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6035850dcc70518ca32f012e46015b9beeda49d8"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.11+0"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "34d526d318358a859d7de23da945578e8e8727b7"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.4+0"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[deps.Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8fdda4c692503d44d04a0603d9ac0982054635f9"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.1+0"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "b4bfde5d5b652e22b9c790ad00af08b6d042b97d"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.15.0+0"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "730eeca102434283c50ccf7d1ecdadf521a765a4"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.2+0"

[[deps.Xorg_xcb_util_cursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_jll", "Xorg_xcb_util_renderutil_jll"]
git-tree-sha1 = "04341cb870f29dcd5e39055f895c39d016e18ccd"
uuid = "e920d4aa-a673-5f3a-b3d7-f755a4d47c43"
version = "0.1.4+0"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "330f955bc41bb8f5270a369c473fc4a5a4e4d3cb"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.6+0"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "691634e5453ad362044e2ad653e79f3ee3bb98c3"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.39.0+0"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e92a1a012a10506618f10b7047e478403a046c77"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.5.0+0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+0"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "49ce682769cd5de6c72dcf1b94ed7790cd08974c"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.5+0"

[[deps.eudev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "gperf_jll"]
git-tree-sha1 = "431b678a28ebb559d224c0b6b6d01afce87c51ba"
uuid = "35ca27e7-8b34-5b7f-bca9-bdc33f59eb06"
version = "3.2.9+0"

[[deps.fzf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "868e669ccb12ba16eaf50cb2957ee2ff61261c56"
uuid = "214eeab7-80f7-51ab-84ad-2988db7cef09"
version = "0.29.0+0"

[[deps.gperf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3516a5630f741c9eecb3720b1ec9d8edc3ecc033"
uuid = "1a1c6b14-54f6-533d-8383-74cd7377aa70"
version = "3.1.1+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3a2ea60308f0996d26f1e5354e10c24e9ef905d4"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.4.0+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.8.0+0"

[[deps.libevdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "141fe65dc3efabb0b1d5ba74e91f6ad26f84cc22"
uuid = "2db6ffa8-e38f-5e21-84af-90c45d0032cc"
version = "1.11.0+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[deps.libinput_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "eudev_jll", "libevdev_jll", "mtdev_jll"]
git-tree-sha1 = "ad50e5b90f222cfe78aa3d5183a20a12de1322ce"
uuid = "36db933b-70db-51c0-b978-0f229ee0e533"
version = "1.18.0+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "b910cb81ef3fe6e78bf6acee440bda86fd6ae00c"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+1"

[[deps.mtdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "814e154bdb7be91d78b6802843f76b6ece642f11"
uuid = "009596ad-96f7-51b1-9f1b-5ce2d5e8a71e"
version = "1.1.6+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.48.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+0"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "9c304562909ab2bab0262639bd4f444d7bc2be37"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.4.1+1"
"""

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
