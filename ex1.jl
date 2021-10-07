### A Pluto.jl notebook ###
# v0.16.1

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
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

	# Code for our model
	ModelSpectrum = ingredients("./src/model_spectrum.jl")
	import .ModelSpectrum:AbstractSpectrum, SimulatedSpectrum, ConvolvedSpectrum, GaussianMixtureConvolutionKernel, doppler_shifted_spectrum
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
If you're using the ICDS-ACI portal to access the Jupyter notebook server, then you need to request that multiple processor cores be allocated to your session when you first submit the request for the JupyterLab server using the box labeled "Number of Cores", i.e. before you open this notebook and even before you start your Pluto session.
"""

# ╔═╡ f76f329a-8dde-4790-96f2-ade735643aeb
if haskey(ENV,"PBS_NUM_PPN")
	pbs_procs_per_node = parse(Int64,ENV["PBS_NUM_PPN"])
	md"Your PBS job was allocated $pbs_procs_per_node cores per node."
end

# ╔═╡ 0e4d7808-47e2-4740-ab93-5d3973eecaa8
if haskey(ENV,"PBS_NUM_PPN")
	if pbs_procs_per_node > 4
		warning_box(md"While we're in class (and the afternoon/evening before labs are due), please ask for just 4 cores, so there will be enough to go around.

		If you return to working on the lab outside of class, then feel free to try benchmarking the code using 8 cores or even 16 cores. Anytime you ask for several cores, then please be extra diligent about closing your session when you're done.")
	end
end

# ╔═╡ 8a50e9fa-031c-4912-8a2d-466e6a9a9935
md"""
This notebook is using **$(Threads.nthreads()) threads**.
"""

# ╔═╡ 7df5fc86-889f-4a5e-ac2b-8c6f68d7c32e
warning_box(md"""
Even when you have a JupyterLab server (or remote desktop or PBS job) that has been allocated multiple cores, that doesn't mean that any code will make use of more than one core.  The ICDS-ACI Portal's Pluto server has been configured to start notebooks with as many threads as physical cores that were allocated to the parent job.

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
cpucores()   # query number of physcal cores

# ╔═╡ f97f1815-50a2-46a9-ac20-e4a3e34d898c
cputhreads() # query number of logical cores

# ╔═╡ 53da8d7a-8620-4fe5-81ba-f615d2d4ed2a
if cpucores() < cputhreads()
	warning_box(md"Your processor is presenting itself as having $(cputhreads()) cores, when it really only has $(cpucores()) cores.  Make sure to limit the number of threads to $(cpucores()).")
end

# ╔═╡ cc1418c8-3261-4c70-bc19-2921695570a6
Threads.nthreads()  # Number of threads avaliable to this Pluto notebook

# ╔═╡ 7f724449-e90e-4f8b-b13c-9640a498893c
@test 1 <= Threads.nthreads() <= cpucores()

# ╔═╡ c85e51b2-2d3d-46a2-8f3f-03b289cab288
 @test !haskey(ENV,"PBS_NUM_PPN") || 1 <= Threads.nthreads() <= pbs_procs_per_node

# ╔═╡ 907766c5-f084-4ddc-bb52-336cb037d521
md"1a.  How many threads is your notebook using?  (Please enter it as an integer rather than a function call, so that it gets stored in your notebook.  That way Matthias and I will be able to interpret the speed-up factors you get below.)"

# ╔═╡ 0bcde4df-1e31-4774-a31f-bd451bb6f758
response_1a = missing # Insert response as simple integer, and not as a variable for function

# ╔═╡ c41d65e3-ea35-4f97-90a1-bfeaeaf927ad
begin
    if !@isdefined(response_1a)
		var_not_defined(:response_1a)
    elseif ismissing(response_1a)
    	still_missing()
	elseif !(typeof(response_1a) <: Integer)
		warning_box(md"response_1a should be an Integer")
	elseif !(1<(response_1a))
		warning_box(md"Please restart your Pluto session at least 2 cores.")
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

In the Helper code section at the bottom of the notebook, we read the code in `src/ModelSpectrum.jl` and place it in a module named ModelSpectrum.  Note that this implicitly includes the code from other files: `continuum.jl`, `spectrum.jl` and `convolution_kernels.jl`.
Then we'll bring several of the custom types into scope, so we can use them easily below.
"""

# ╔═╡ c31cf36c-21ec-46f1-96aa-b014ff094f8a
md"""
## Synthetic Spectrum
In this exercise, we're going to create a model spectrum consisting of continuum, stellar absorption lines, telluric absorption lines.
The `ModelSpectrum` module provides a `SimulatedSpectrum` type.
We need to create a `SimulatedSpectrum` object that contains specific parameter values.  The function below will do that for us.
"""

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

# ╔═╡ 86b8dd31-1261-4fb9-bfd3-13f6f01e7790
# Create a functor (function object) that computes a model spectrum that we'll analyze below
raw_spectrum = make_spectrum_object(lambda_min=lambda_min,lambda_max=lambda_max)

# ╔═╡ 16ad0225-c7d6-455b-8eb0-3e93c9f9f91a
md"## Convolved spectrum

Next, we will create an object containing a model for the point spread function (implemented as a mixture of multiple Gaussians).
Then we create a funtor that can compute the convolution of our spectral model with the point spread function model.
"

# ╔═╡ 65398796-73ab-4d98-9851-3bb162ac8cbc
begin      # Create a model for the point spread function (PSF)
	psf_widths  = [0.5, 1.0, 2.0]
	psf_weights = [0.8, 0.15, 0.05]
	psf_model = GaussianMixtureConvolutionKernel(psf_widths,psf_weights)
end

# ╔═╡ 0aafec61-ff44-49e2-95e9-d3506ac6afa7
# Create a functor (function object) that computes a model for the the convolution of the raw spectrum with the PSF model
conv_spectrum = ConvolvedSpectrum(raw_spectrum,psf_model)

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

# ╔═╡ dbf05374-1d89-4f30-b4b4-6cf57631f8b7
begin
	plot(lambdas,raw_spectrum.(lambdas),xlabel="λ", ylabel="Flux", label="Raw spectrum", legend=:bottomright)
	plot!(lambdas,conv_spectrum.(lambdas), label="Convolved spectrum")
end

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

# ╔═╡ f2b23082-98bc-4be1-bb6d-cac8facb8a46
let
	plt = plot(view(lambdas,idx_plot),raw_spectrum.(view(lambdas,idx_plot)),xlabel="λ", ylabel="Flux", label="Raw spectrum", legend=:bottomright)
	plot!(plt,view(lambdas,idx_plot),conv_spectrum.(view(lambdas,idx_plot)), label="Convolved spectrum")
	ylims!(plt,0,1.01)
end

# ╔═╡ ee96411d-e3fa-442b-b0fe-10d6ede37b6a
md"""
You can adjust the sliders to interactively explore our model spectra.
"""

# ╔═╡ b92aad2e-8a3b-4edf-ae7e-6e3cff6eead4
protip(md"Feel free to look at the hidden code in the cells above for the lower plot and slider bars, as well as the documentation at [PlutoUI.jl](https://juliahub.com/docs/PlutoUI/abXFp/0.7.9/autodocs/) for examples of how to make interactive widgets in your notebooks.")

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

# ╔═╡ d6d3a2d1-241e-44c1-a11b-5bfb2b3c5f4b
md"""
As expected, the different versions perform very similarly in terms of wall-clock time and memory allocated.
"""

# ╔═╡ 0344a74d-456b-44f0-84dc-c2fdbd41a379
md"""
## Benchmarking convolved spectrum

Next, we'll evaluate the convolution of the raw spectrum with the PDF model at each of the wavelengths, using `conv_spectrum`.
"""

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

In principle, we could further optimize the serial version to avoid unnecessary memory allocations by diving into the internals of `quadgk` and preallocating a workspace for the calculations.  However, the code in the QuadGk.jl package isn't the easiest to read, and we'd have to spend considerable time understanding, writing and testing such changes.  In practice, it's often a better use of our time to parallelize a pretty efficient serial code, rather than writing the most efficient serial code possible.")

# ╔═╡ db1583f4-61cb-43e0-9326-d6c15d8fad5a
md"""
## Map
Our calculation is one example of a very useful programming pattern, known as **map**.  The map pattern corresponds to problems where the total work can be organized as doing one smaller calculation many times with different input values.
Julia provides a [`map`](https://docs.julialang.org/en/v1/base/collections/#Base.map) function (as well as `map!` for writing to memory that's been preallocated ) that can be quite useful.
`map(func,collection)` applies func to every element of the collection and returns a collection similar in size to collection.
In our example, each input wavelength is mapped to out output flux.
"""

# ╔═╡ ca9c7d9e-e6cc-46cc-8a9b-ccda123591a2
if true
	map(raw_spectrum,lambdas)
	stats_map_serial_raw = @timed map(raw_spectrum,lambdas)
	(;  time=stats_map_serial_raw.time, bytes=stats_map_serial_raw.bytes)
end

# ╔═╡ 215011e0-5977-43f8-bb65-83c09b3c07d8
if true
	result_spec_serial_map = map(conv_spectrum,lambdas)
	stats_spec_serial_map = @timed map(conv_spectrum,lambdas)
	(;  time=stats_spec_serial_map.time, bytes=stats_spec_serial_map.bytes )
end

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
protip(md"
It is possible to have each function return an array (or NamedTuple or custom struct).  Then the output is an array of arrays (or array of NamedTuples or array of structs).  However, getting the outputs in the format we want to use for subsequent calculations (e.g., arrays for each output, rather than an array of structs) is often more tedious and error prone than just writing the for loop version.")

# ╔═╡ a44a3478-541d-40d6-9d99-04b918c16bfb
md"""We'll implement a serial version as a starting point and comparison.
"""

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

# ╔═╡ 96914ff8-56c8-4cc8-96bc-fd3d13f7e4ce
md"As expected the performance is very similar to the broadcasted for mappeed version."

# ╔═╡ 32685a28-54d9-4c0d-8940-e82843d2cab2
md"# Parallelization via multiple threads"

# ╔═╡ 3717d201-0bc3-4e3c-8ecd-d835e58f6821
md"""
Julia has native support for using multiple **threads**.  This is useful when you have one computer with multiple processor cores.  Then each thread can execute on a separate processor core.  Because the threads are part of the same **process**, every thread has access to all the memory used by every other thread.  Programming with threads requires being careful to avoid undefined behavior because threads read and write to the same memory location in an unexpected order.  In a general multi-threaded programming can be intimidating, since arbitrary parallel code is hard to write, read, debug and maintain.  One way to keep things managable is to stick with some common programming patterns which are relatively easy to work with.  We'll explore using threads for a parallel for and a parallel map.
"""

# ╔═╡ 496e8c5e-251b-4448-8c59-541877d752c1
md"""
## Parallel Map

If you can write your computations in terms of calling `map`, then one easy way to parallelize your code is to replace the call to `map` with a call to `ThreadsX.map`, a parallel map that makes use of multiple threads.
If your julia kernel has only a single thread, then it will still run in serial.  But if you have multiple theads, then `ThreadsX.map` will parallelize your code.
"""

# ╔═╡ 04bcafcd-1d2f-4ce5-893f-7ec5bb05f9ed
md"""
1a.  Given that this notebook is using $(Threads.nthreads()) threads, what is the theoretical maximum improvement in performance?  How much faster do you expect the `conv_spectrum` code to run using `ThreadsX.map` relative to searial `map`?
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

# ╔═╡ c7121d63-b1ff-4c38-8579-e1adbfef48ef
if !ismissing(response_1b)
	result_spec_ThreadsXmap = ThreadsX.map(conv_spectrum,lambdas)
	stats_spec_ThreadsXmap = @timed ThreadsX.map(conv_spectrum,lambdas)
	(;  time=stats_spec_ThreadsXmap.time,
		bytes=stats_spec_ThreadsXmap.bytes )
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
You were likely disappointed in the speed-up factor.  What could have gone wrong?  In this case, we have a non-trivial, but still modest amount of work to do for each wavelength.  `map` distributed the work one element at a time.  The overhead in distributing the work and assembling the pieces likely ate into the potential performance gains.  To improve on this, we can tell `map` to distribute the work in batches.  Below, we'll specify an optional named parameter, `basesize`.
"""

# ╔═╡ fb063bc5-22bc-4b32-8fcb-5fbc4765c8b5
batchsize_for_ThreadsXmap = 256

# ╔═╡ 0e9664ec-98d8-49d4-a376-24d4770c4c8f
if !ismissing(response_1c)
	ThreadsX.map(conv_spectrum,lambdas,basesize=batchsize_for_ThreadsXmap)
	walltime_ThreadsXmap_batched = @elapsed ThreadsX.map(conv_spectrum,lambdas,basesize=batchsize_for_ThreadsXmap)
end

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

# ╔═╡ e55c802d-7923-458f-af42-d951e82e029b
function calc_spectrum_threaded_for_loop(x::AbstractArray, spectrum::T) where T<:AbstractSpectrum
    out = zeros(length(x))
    Threads.@threads for i in 1:length(x)
        @inbounds out[i] = spectrum(x[i])
    end
    return out
end

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

# ╔═╡ b3a6004f-9d10-4582-832a-8917b701f2ad
if !ismissing(response_1e)
	result_spec_threaded_loop = calc_spectrum_threaded_for_loop(lambdas,conv_spectrum)
	stats_spec_threaded_loop = @timed calc_spectrum_threaded_for_loop(lambdas,conv_spectrum)
	(;  time=stats_spec_threaded_loop.time,
		bytes=stats_spec_threaded_loop.bytes )
end

# ╔═╡ 791041e9-d277-4cac-a5ac-1d6ec52e0287
md"""
While Threads.@threads can be useful for some simple tasks, there is active development of packages that provide additional features for multi-threaded programming.  For example, the ThreadsX package provides a `foreach` function and the FLoops package provides a `@floop` macro, both of which we'll demonstrate and benchmark below.
"""

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

# ╔═╡ d43525da-e0a2-4d2f-9dbb-bf187eebf6c1
tip(md"""
## ''Embarassingly'' parallel is good

So far, we've demonstrated parallelizing a computation that can be easily broken into smaller tasks that do not need to communicate with each other.  This is often called an called *embarassingly parallel* computation.  Don't let the name mislead you.  While it could be embarassingly if a Computer Science graduate student tried to make a Ph.D. thesis out of parallelizing an embarassingly parallel problem, that doesn't mean that programmers shouldn't take advantage of opportunities to use embarssingly parallel techniques when they can.If you can parallelize your code using embarassingly parallel techniques, then you should almost always parallelize it that way, instead of (or at least before) trying to parallelize it at a finer grained level.

Next, we'll consider problems that do require some communications between tasks, but in a very structured manner.
""")

# ╔═╡ 547ad5ba-06ad-4707-a7ef-e444cf88ae53
md"""
# Reductions
Many common calculations can be formulated as a [**reduction operation**](https://en.wikipedia.org/wiki/Reduction_operator), where many inputs are transformed into one output.  Common examples would be `sum` or `maximum`.  One key property of reduction operations is that they are associative, meaning it's ok for the computer to change the order in which inputs are reduced.  (Thinking back to our lesson about floating point arithmetic, many operation aren't formally associative or commutative, but are still close enough that we're willing to let the computer reorder calculations.)

When we have multiple processors, the input can be divided into subsets and each processor reduce each subset separately.  Then each processor only needs to communicate one value of the variable being reduced to another processor, even if the input is quite large.  For some problems, reductions also reduce the ammount of memory allocations necessary.
"""

# ╔═╡ 7ba35a63-ac61-434b-b759-95d505f62d9e
md"""
We'll explore different ways to perform reductions on an example problem where we calculate the mean squared error between the model and the model Doppler shifted by a velocity, $v$. First, let's write a vanilla serial version, where we first compute an array of squared residuals and pass that to the `sum` function.
"""

# ╔═╡ 398ba928-899f-4843-ad58-25df67c81ffe
function calc_mse_broadcasted(lambdas::AbstractArray, spec1::AbstractSpectrum, spec2::AbstractSpectrum, v::Number)
	c = ModelSpectrum.speed_of_light
	z = v/c
	spec2_shifted = doppler_shifted_spectrum(spec2,z)
	mse = sum((spec1.(lambdas) .- spec2_shifted.(lambdas)).^2)
	mse /= length(lambdas)
end

# ╔═╡ cee9c93d-cf7b-4da1-b4bb-b544b7cc104c
v = 10.0

# ╔═╡ 9f8667f3-4104-4642-b2d9-a6d12a6fa5d3
begin
	result_mse_broadcasted = calc_mse_broadcasted(lambdas,conv_spectrum,conv_spectrum,v)
	stats_mse_broadcasted = @timed calc_mse_broadcasted(lambdas,conv_spectrum,conv_spectrum,v)
end

# ╔═╡ 3ac01c04-52e3-497e-8c29-8c704e23ae39
md"## Serial loop with reduction"

# ╔═╡ 790377a7-1301-44a8-b300-418567737373
md"""
Now we'll write a version of the function using a serial for loop.  Note that we no longer need to allocate an output array, since `calc_mse_loop` only needs to return the reduced mean squared error and not the value of the spectrum at every wavelength.
"""

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
As expected, the $(floor(Int,stats_mse_loop.bytes//1024^2)) MB allocated when we compute the mean squared error between *two* simulated spectra is very nearly twice the the $(floor(Int,stats_spec_serial_loop.bytes//1024^2)) MB allocated by the serial for loop to compute the one spectrum at each wavelength.
"""

# ╔═╡ e36cda69-d300-4156-9bef-a372f94306d9
md"""
Similarly, it's likely that the wal time for the serial loop to compute the mean squared error $(stats_mse_loop.time) sec
is nearly twice that of the serial loop to compute one spectrum $(stats_spec_serial_loop.time) sec.
So far it doesn't seem particularly interesting.
"""

# ╔═╡ 161ea6af-5661-44e1-ae40-1b581b636c25
md"""
## Parallel loop with reduction
Next, we'll use [FLoops.jl](https://github.com/JuliaFolds/FLoops.jl) to compute the mean sequared error using multiple threads.  Note the we need to use the `@floop` macro around the loop  *and* the `@reduce` macro to indicate which variables are part of the reduction.
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
One advantage of parallelizing your code with [FLoops.jl](https://juliafolds.github.io/FLoops.jl/dev/) is that it then becomes very easy to compare the performance of a calculation in serial and in parallel using different **[executors](https://juliafolds.github.io/FLoops.jl/dev/tutorials/parallel/#tutorials-executor)** that specify how the calculation should be implemented.  There are different parallel executor for shared-memory parallelism (via multi-threading this exercise), distributed-memory parallelism (see [Lab 7](https://github.com/PsuAstro528/lab7-start)) and even for parallelizing code over a GPUs (although there are some restrictions on what code can be run on the GPU, that we'll see in a [Lab 8](https://github.com/PsuAstro528/lab3-start)).
"""

# ╔═╡ 383aa611-e115-482e-873c-4487e53d457f
md"# Mapreduce

We can combine `map` and `reduce` into one function `mapreduce`.  There are opportunities for some increased efficiencies when merging the two, since the ammount of communications between threads can be significantly decreased thanks to the reduction operator.  Mapreduce is a common, powerful and efficient programming pattern.  For example, we often want to evaluate a model for many input values, compare the results of the model to data and the compute some statistic about how much the model and data differ.

In this exercise, we'll demonstrate using `mapreduce` for calculating the mean squared error between the model and the model Doppler shifted by a velocity, $v$.  First, we'll
"

# ╔═╡ 2c6fa743-3dec-417b-b05a-17bb52b5d39d
 md"## Mapreduce (serial)"

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

# ╔═╡ ae47ef38-e8d0-40b9-9e61-3ab3ca7e7a49
md"## Parallel mapreduce"

# ╔═╡ aad94861-e2b3-417d-b640-b821e53adb23
md"""
The ThreasX package provides a multi-threaded version of mapreduce that we can easily drop in.
"""

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
	result_mse_mapreduce_threadsx_bs16 = calc_mse_mapreduce_threadsx(lambdas,conv_spectrum,conv_spectrum,v; basesize=16)
	stats_mse_mapreduce_threadsx_bs16 = @timed calc_mse_mapreduce_threadsx(lambdas,conv_spectrum,conv_spectrum,v; basesize=16)
	(;  time=stats_mse_mapreduce_threadsx_bs16.time, bytes=stats_mse_mapreduce_threadsx_bs16.bytes )
end

# ╔═╡ 3f01d534-b01d-4ab4-b3cd-e809b02563a9
md"""
1h.  How did the performance of `calc_mse_mapreduce_threadsx` compare to the performance of `calc_mse_map_serial`?  Can you explain why this differs from the comparison of `calc_spectrum_mapreduce_threadsx` to `ThreadsX.map(conv_spectrum,lambdas,..)`?
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
1i.  Think about how you will parallelize your class project code.  The first parallelization uses a shared-memory model.  Which of these programming patterns would be a good fit for your project?  Can your project calculation be formulated as a map or mapreduce problem?  If not, then could it be implemented as a series of multiple maps/reductions/mapreduces?

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

# ╔═╡ 8737797c-6563-4513-a5fc-fde9681b4c63
md"""
1j.  Before parallelizing your project code for shared memory, try parallelizing , get some practice by writing a function calc_χ²_my_way in the cell below to parallelize the calculation of calculating χ² using whichever parallelization strategy that you plan to use for your project.  Feel free to refer to the serial function function `calc_χ²` at bottom of notebook.
"""

# ╔═╡ 87df5b25-0d2f-4f81-80f1-aaf6c9f89ce3
# response_1i:
function calc_χ²_my_way(lambdas::AbstractArray, spec1::AbstractSpectrum, spec2::AbstractSpectrum, σ1::AbstractArray, σ2::AbstractArray, v::Number; #= any optional parameters? =# )
    # INSERT YOUR CODE HERE
    return missing
end

# ╔═╡ bd77bc71-ffdf-4ba1-b1ee-6f2a69044e6f
begin
    σ_obs1 = 0.02*ones(size(lambdas))
    σ_obs2 = 0.02*ones(size(lambdas))
end;

# ╔═╡ 4dec4888-08db-4965-b27a-fc44f316b529
begin
    if !@isdefined(calc_χ²_my_way)
		func_not_defined(:calc_χ²_my_way)
    elseif ismissing(calc_χ²_my_way(lambdas,conv_spectrum, conv_spectrum, σ_obs1, σ_obs2, 0.0))
    	still_missing()
	end
end

# ╔═╡ 6f411bcc-7084-43c3-a88b-b56ba77b5732
begin
    calc_χ²_my_way, lambdas, conv_spectrum, σ_obs1, σ_obs2
    @test abs(calc_χ²_my_way(lambdas,conv_spectrum, conv_spectrum, σ_obs1, σ_obs2, 0.0 )) < 1e-8
end

# ╔═╡ 3b50062c-99c1-4f68-aabe-2d40d4ad7504
md"## Helper code"

# ╔═╡ d83a282e-cb2b-4837-bfd4-8404b3722e3a
ChooseDisplayMode()

# ╔═╡ c9cf6fb3-0146-42e6-aaae-24e97254c805
TableOfContents(aside=true)

# ╔═╡ a9601654-8263-425e-8d8f-c5bbeacbbe06
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
BenchmarkTools = "~1.2.0"
CpuId = "~0.3.0"
Distributions = "~0.25.16"
FLoops = "~0.1.11"
Plots = "~1.22.1"
PlutoTeachingTools = "~0.1.4"
PlutoTest = "~0.1.1"
PlutoUI = "~0.7.10"
QuadGK = "~2.4.2"
StaticArrays = "~1.2.12"
ThreadsX = "~0.1.8"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

[[Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "84918055d15b3114ede17ac6a7182f68870c16f7"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.3.1"

[[ArgCheck]]
git-tree-sha1 = "dedbbb2ddb876f899585c4ec4433265e3017215a"
uuid = "dce04be8-c92d-5529-be00-80e4d2c0e197"
version = "2.1.0"

[[ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[BangBang]]
deps = ["Compat", "ConstructionBase", "Future", "InitialValues", "LinearAlgebra", "Requires", "Setfield", "Tables", "ZygoteRules"]
git-tree-sha1 = "0ad226aa72d8671f20d0316e03028f0ba1624307"
uuid = "198e06fe-97b7-11e9-32a5-e1d131e6ad66"
version = "0.3.32"

[[Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[Baselet]]
git-tree-sha1 = "aebf55e6d7795e02ca500a689d326ac979aaf89e"
uuid = "9718e550-a3fa-408a-8086-8db961cd8217"
version = "0.1.1"

[[BenchmarkTools]]
deps = ["JSON", "Logging", "Printf", "Profile", "Statistics", "UUIDs"]
git-tree-sha1 = "61adeb0823084487000600ef8b1c00cc2474cd47"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.2.0"

[[Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "f2202b55d816427cd385a9a4f3ffb226bee80f99"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+0"

[[ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "bd4afa1fdeec0c8b89dad3c6e92bc6e3b0fec9ce"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.6.0"

[[ColorSchemes]]
deps = ["ColorTypes", "Colors", "FixedPointNumbers", "Random"]
git-tree-sha1 = "9995eb3977fbf67b86d0a0a0508e83017ded03f2"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.14.0"

[[ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "024fe24d83e4a5bf5fc80501a314ce0d1aa35597"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.0"

[[Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "417b0ed7b8b838aa6ca0a87aadf1bb9eb111ce40"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.8"

[[Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "4866e381721b30fac8dda4c8cb1d9db45c8d2994"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.37.0"

[[CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[CompositionsBase]]
git-tree-sha1 = "455419f7e328a1a2493cabc6428d79e951349769"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.1"

[[ConstructionBase]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f74e9d5388b8620b4cee35d4c5a618dd4dc547f4"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.3.0"

[[ContextVariablesX]]
deps = ["Compat", "Logging", "UUIDs"]
git-tree-sha1 = "8ccaa8c655bc1b83d2da4d569c9b28254ababd6e"
uuid = "6add18c4-b38d-439d-96f6-d6bc489c04c5"
version = "0.1.2"

[[Contour]]
deps = ["StaticArrays"]
git-tree-sha1 = "9f02045d934dc030edad45944ea80dbd1f0ebea7"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.5.7"

[[CpuId]]
deps = ["Markdown"]
git-tree-sha1 = "32d125af0fb8ec3f8935896122c5e345709909e5"
uuid = "adafc99b-e345-5852-983c-f28acb93d879"
version = "0.3.0"

[[DataAPI]]
git-tree-sha1 = "cc70b17275652eb47bc9e5f81635981f13cea5c8"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.9.0"

[[DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "7d9d316f04214f7efdbb6398d545446e246eff02"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.10"

[[DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[DefineSingletons]]
git-tree-sha1 = "77b4ca280084423b728662fe040e5ff8819347c5"
uuid = "244e2a9f-e319-4986-a169-4d1fe445cd52"
version = "0.1.1"

[[DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[Distributions]]
deps = ["ChainRulesCore", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns"]
git-tree-sha1 = "f4efaa4b5157e0cdb8283ae0b5428bc9208436ed"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.16"

[[DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "a32185f5428d3986f47c2ab78b1f216d5e6cc96f"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.5"

[[Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3f3a2501fa7236e9b911e0f7a588c657e822bb6d"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.3+0"

[[Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b3bfd02e98aedfa5cf885665493c5598c350cd2f"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.2.10+0"

[[FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "Pkg", "Zlib_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "d8a578692e3077ac998b50c0217dfd67f21d1e5f"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.0+0"

[[FLoops]]
deps = ["Compat", "FLoopsBase", "JuliaVariables", "MLStyle", "Serialization", "Setfield", "Transducers"]
git-tree-sha1 = "7cb2eb7e5d824885a4d5e0a7870660c01ac394c2"
uuid = "cc61a311-1640-44b5-9fba-1b764f453329"
version = "0.1.11"

[[FLoopsBase]]
deps = ["ContextVariablesX"]
git-tree-sha1 = "cf3d8b2527be12d204d06aba922b30339a9653dd"
uuid = "b9860ae5-e623-471e-878b-f6a53c775ea6"
version = "0.1.0"

[[FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "caf289224e622f518c9dbfe832cdafa17d7c80a6"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.12.4"

[[FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "87eb71354d8ec1a96d4a7636bd57a7347dde3ef9"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.10.4+0"

[[FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "dba1e8614e98949abfa60480b13653813d8f0157"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.5+0"

[[GR]]
deps = ["Base64", "DelimitedFiles", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Printf", "Random", "Serialization", "Sockets", "Test", "UUIDs"]
git-tree-sha1 = "c2178cfbc0a5a552e16d097fae508f2024de61a3"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.59.0"

[[GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Pkg", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "ef49a187604f865f4708c90e3f431890724e9012"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.59.0+0"

[[GeometryBasics]]
deps = ["EarCut_jll", "IterTools", "LinearAlgebra", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "58bcdf5ebc057b085e58d95c138725628dd7453c"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.4.1"

[[Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "7bf67e9a481712b3dbe9cb3dac852dc4b1162e02"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.68.3+0"

[[Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[HTTP]]
deps = ["Base64", "Dates", "IniFile", "Logging", "MbedTLS", "NetworkOptions", "Sockets", "URIs"]
git-tree-sha1 = "60ed5f1643927479f845b0135bb369b031b541fa"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "0.9.14"

[[HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "8a954fed8ac097d5be04921d595f741115c1b2ad"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+0"

[[HypertextLiteral]]
git-tree-sha1 = "1e3ccdc7a6f7b577623028e0095479f4727d8ec1"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.8.0"

[[IniFile]]
deps = ["Test"]
git-tree-sha1 = "098e4d2c533924c921f9f9847274f2ad89e018b8"
uuid = "83e8ac13-25f8-5344-8a64-a9f2b223428f"
version = "0.5.0"

[[InitialValues]]
git-tree-sha1 = "26c8832afd63ac558b98a823265856670d898b6c"
uuid = "22cec73e-a1b8-11e9-2c92-598750a2cf9c"
version = "0.2.10"

[[InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[IrrationalConstants]]
git-tree-sha1 = "f76424439413893a832026ca355fe273e93bce94"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.0"

[[IterTools]]
git-tree-sha1 = "05110a2ab1fc5f932622ffea2a003221f4782c18"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.3.0"

[[IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "642a199af8b68253517b80bd3bfd17eb4e84df6e"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.3.0"

[[JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "8076680b162ada2a031f707ac7b4953e30667a37"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.2"

[[JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "d735490ac75c5cb9f1b00d8b5509c11984dc6943"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.0+0"

[[JuliaVariables]]
deps = ["MLStyle", "NameResolution"]
git-tree-sha1 = "49fb3cb53362ddadb4415e9b73926d6b40709e70"
uuid = "b14d175d-62b4-44ba-8fb7-3064adc8c3ec"
version = "0.2.4"

[[LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[LaTeXStrings]]
git-tree-sha1 = "c7f1c695e06c01b95a67f0cd1d34994f3e7db104"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.2.1"

[[Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "Printf", "Requires"]
git-tree-sha1 = "a4b12a1bd2ebade87891ab7e36fdbce582301a92"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.6"

[[LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "761a393aeccd6aa92ec3515e428c26bf99575b3b"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+0"

[[Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "7739f837d6447403596a75d19ed01fd08d6f56bf"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.3.0+3"

[[Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "42b62845d70a619f063a7da093d995ec8e15e778"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.16.1+1"

[[Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "340e257aada13f95f98ee352d316c3bed37c8ab9"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.3.0+0"

[[Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[LinearAlgebra]]
deps = ["Libdl"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[LogExpFunctions]]
deps = ["ChainRulesCore", "DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "34dc30f868e368f8a17b728a1238f3fcda43931a"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.3"

[[Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[MLStyle]]
git-tree-sha1 = "594e189325f66e23a8818e5beb11c43bb0141bcd"
uuid = "d8e11817-5142-5d16-987a-aa16d5891078"
version = "0.4.10"

[[MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "5a5bc6bf062f0f95e62d0fe0a2d99699fed82dd9"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.8"

[[Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "Random", "Sockets"]
git-tree-sha1 = "1c38e51c3d08ef2278062ebceade0e46cefc96fe"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.0.3"

[[MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[Measures]]
git-tree-sha1 = "e498ddeee6f9fdb4551ce855a46f54dbd900245f"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.1"

[[MicroCollections]]
deps = ["BangBang", "Setfield"]
git-tree-sha1 = "4f65bdbbe93475f6ff9ea6969b21532f88d359be"
uuid = "128add7d-3638-4c79-886c-908ea0c25c34"
version = "0.1.1"

[[Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "bf210ce90b6c9eed32d25dbcae1ebc565df2687f"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.2"

[[Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[NaNMath]]
git-tree-sha1 = "bfe47e760d60b82b66b61d2d44128b62e3a369fb"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "0.3.5"

[[NameResolution]]
deps = ["PrettyPrint"]
git-tree-sha1 = "1a0fa0e9613f46c9b8c11eee38ebb4f590013c5e"
uuid = "71a1bf82-56d0-4bbc-8a3c-48b961074391"
version = "0.1.5"

[[NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7937eda4681660b4d6aeeecc2f7e1c81c8ee4e2f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+0"

[[OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "15003dcb7d8db3c6c857fda14891a539a8f2705a"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.10+0"

[[OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[PCRE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b2a7af664e098055a7529ad1a900ded962bca488"
uuid = "2f80f16e-611a-54ab-bc61-aa92de5b98fc"
version = "8.44.0+0"

[[PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "4dd403333bcf0909341cfe57ec115152f937d7d8"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.1"

[[Parsers]]
deps = ["Dates"]
git-tree-sha1 = "438d35d2d95ae2c5e8780b330592b6de8494e779"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.0.3"

[[Pixman_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b4f5d02549a10e20780a24fce72bea96b6329e29"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.40.1+0"

[[Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[PlotThemes]]
deps = ["PlotUtils", "Requires", "Statistics"]
git-tree-sha1 = "a3a964ce9dc7898193536002a6dd892b1b5a6f1d"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "2.0.1"

[[PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "2537ed3c0ed5e03896927187f5f2ee6a4ab342db"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.0.14"

[[Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "GeometryBasics", "JSON", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "PlotThemes", "PlotUtils", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs"]
git-tree-sha1 = "4c2637482176b1c2fb99af4d83cb2ff0328fc33c"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.22.1"

[[PlutoTeachingTools]]
deps = ["LaTeXStrings", "Markdown", "PlutoUI", "Random"]
git-tree-sha1 = "e2b63ee022e0b20f43fcd15cda3a9047f449e3b4"
uuid = "661c6b06-c737-4d37-b85c-46df65de6f69"
version = "0.1.4"

[[PlutoTest]]
deps = ["HypertextLiteral", "InteractiveUtils", "Markdown", "Test"]
git-tree-sha1 = "ada2eae88798ed6c93d9acb5e41e1671794bb8c8"
uuid = "cb4044da-4d16-4ffa-a6a3-8cad7f73ebdc"
version = "0.1.1"

[[PlutoUI]]
deps = ["Base64", "Dates", "HypertextLiteral", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "Suppressor"]
git-tree-sha1 = "26b4d16873562469a0a1e6ae41d90dec9e51286d"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.10"

[[Preferences]]
deps = ["TOML"]
git-tree-sha1 = "00cfd92944ca9c760982747e9a1d0d5d86ab1e5a"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.2.2"

[[PrettyPrint]]
git-tree-sha1 = "632eb4abab3449ab30c5e1afaa874f0b98b586e4"
uuid = "8162dcfd-2161-5ef2-ae6c-7681170c5f98"
version = "0.2.0"

[[Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[Profile]]
deps = ["Printf"]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"

[[Qt5Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "xkbcommon_jll"]
git-tree-sha1 = "ad368663a5e20dbb8d6dc2fddeefe4dae0781ae8"
uuid = "ea2cea3b-5b76-57ae-a6ef-0a8af62496e1"
version = "5.15.3+0"

[[QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "78aadffb3efd2155af139781b8a8df1ef279ea39"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.4.2"

[[REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[Random]]
deps = ["Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[RecipesBase]]
git-tree-sha1 = "44a75aa7a527910ee3d1751d1f0e4148698add9e"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.1.2"

[[RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "RecipesBase"]
git-tree-sha1 = "7ad0dfa8d03b7bcf8c597f59f5292801730c55b8"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.4.1"

[[Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[Referenceables]]
deps = ["Adapt"]
git-tree-sha1 = "e681d3bfa49cd46c3c161505caddf20f0e62aaa9"
uuid = "42d2dcc6-99eb-4e98-b66c-637b7d73030e"
version = "0.1.2"

[[Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "4036a3bd08ac7e968e27c203d45f5fff15020621"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.1.3"

[[Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "bf3188feca147ce108c76ad82c2792c57abe7b1f"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.0"

[[Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "68db32dff12bb6127bac73c209881191bf0efbb7"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.3.0+0"

[[SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[Scratch]]
deps = ["Dates"]
git-tree-sha1 = "0b4b7f1393cff97c33891da2a0bf69c6ed241fda"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.1.0"

[[Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "Requires"]
git-tree-sha1 = "def0718ddbabeb5476e51e5a43609bee889f285d"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "0.8.0"

[[SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "b3363d7460f7d098ca0912c69b082f75625d7508"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.0.1"

[[SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[SpecialFunctions]]
deps = ["ChainRulesCore", "LogExpFunctions", "OpenSpecFun_jll"]
git-tree-sha1 = "a322a9493e49c5f3a10b50df3aedaf1cdb3244b7"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "1.6.1"

[[SplittablesBase]]
deps = ["Setfield", "Test"]
git-tree-sha1 = "39c9f91521de844bad65049efd4f9223e7ed43f9"
uuid = "171d559e-b47b-412a-8079-5efa626c420e"
version = "0.1.14"

[[StaticArrays]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "3240808c6d463ac46f1c1cd7638375cd22abbccb"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.2.12"

[[Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[StatsAPI]]
git-tree-sha1 = "1958272568dc176a1d881acb797beb909c785510"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.0.0"

[[StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "8cbbc098554648c84f79a463c9ff0fd277144b6c"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.10"

[[StatsFuns]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "46d7ccc7104860c38b11966dd1f72ff042f382e4"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "0.9.10"

[[StructArrays]]
deps = ["Adapt", "DataAPI", "StaticArrays", "Tables"]
git-tree-sha1 = "2ce41e0d042c60ecd131e9fb7154a3bfadbf50d3"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.3"

[[SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[Suppressor]]
git-tree-sha1 = "a819d77f31f83e5792a76081eee1ea6342ab8787"
uuid = "fd094767-a336-5f1f-9728-57cf17d0bbfb"
version = "0.2.0"

[[TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "TableTraits", "Test"]
git-tree-sha1 = "1162ce4a6c4b7e31e0e6b14486a6986951c73be9"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.5.2"

[[Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[ThreadsX]]
deps = ["ArgCheck", "BangBang", "ConstructionBase", "InitialValues", "MicroCollections", "Referenceables", "Setfield", "SplittablesBase", "Transducers"]
git-tree-sha1 = "abcff3ac31c7894550566be533b512f8b059104f"
uuid = "ac1d9e8a-700a-412c-b207-f0111f4b6c0d"
version = "0.1.8"

[[Transducers]]
deps = ["Adapt", "ArgCheck", "BangBang", "Baselet", "CompositionsBase", "DefineSingletons", "Distributed", "InitialValues", "Logging", "Markdown", "MicroCollections", "Requires", "Setfield", "SplittablesBase", "Tables"]
git-tree-sha1 = "dec7b7839f23efe21770b3b1307ca77c13ed631d"
uuid = "28d57a85-8fef-5791-bfe6-a80928e7c999"
version = "0.4.66"

[[URIs]]
git-tree-sha1 = "97bbe755a53fe859669cd907f2d96aee8d2c1355"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.3.0"

[[UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[Wayland_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "3e61f0b86f90dacb0bc0e73a0c5a83f6a8636e23"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.19.0+0"

[[Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll"]
git-tree-sha1 = "2839f1c1296940218e35df0bbb220f2a79686670"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.18.0+4"

[[XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "1acf5bdf07aa0907e0a37d3718bb88d4b687b74a"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.9.12+0"

[[XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "5be649d550f3f4b95308bf0183b82e2582876527"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.6.9+4"

[[Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4e490d5c960c314f33885790ed410ff3a94ce67e"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.9+4"

[[Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fe47bd2247248125c428978740e18a681372dd4"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.3+4"

[[Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

[[Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6783737e45d3c59a4a4c4091f5f88cdcf0908cbb"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.0+3"

[[Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "daf17f441228e7a3833846cd048892861cff16d6"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.13.0+3"

[[Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "926af861744212db0eb001d9e40b5d16292080b2"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.0+4"

[[Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "4bcbf660f6c2e714f87e960a171b119d06ee163b"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.2+4"

[[Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "5c8424f8a67c3f2209646d4425f3d415fee5931d"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.27.0+4"

[[Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "79c31e7844f6ecf779705fbc12146eb190b7d845"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.4.0+3"

[[Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "cc4bf3fdde8b7e3e9fa0351bdeedba1cf3b7f6e6"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.0+0"

[[ZygoteRules]]
deps = ["MacroTools"]
git-tree-sha1 = "9e7a1e8ca60b742e508a315c17eef5211e7fbfd7"
uuid = "700de1a5-db45-46bc-99cf-38207098b444"
version = "0.2.1"

[[libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "c45f4e40e7aafe9d086379e5578947ec8b95a8fb"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+0"

[[nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"

[[x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "ece2350174195bb31de1a63bea3a41ae1aa593b6"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "0.9.1+5"
"""

# ╔═╡ Cell order:
# ╟─85aad005-eac0-4f71-a32c-c8361c31813b
# ╟─bdf61711-36e0-40d5-b0c5-3bac20a25aa3
# ╟─629442ba-a968-4e35-a7cb-d42a0a8783b4
# ╟─0bee1c3c-b130-49f2-baa4-efd8e3b49fdc
# ╠═f76f329a-8dde-4790-96f2-ade735643aeb
# ╠═0e4d7808-47e2-4740-ab93-5d3973eecaa8
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
# ╠═8d7c27d5-4a07-4ab4-9ece-94fdb7053f73
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
# ╟─bd77bc71-ffdf-4ba1-b1ee-6f2a69044e6f
# ╠═6f411bcc-7084-43c3-a88b-b56ba77b5732
# ╠═3c5ee822-b938-4848-b2b0-f0de2e65b4db
# ╟─3b50062c-99c1-4f68-aabe-2d40d4ad7504
# ╟─d83a282e-cb2b-4837-bfd4-8404b3722e3a
# ╟─c9cf6fb3-0146-42e6-aaae-24e97254c805
# ╠═76730d06-06da-4466-8814-2096b221090f
# ╠═a9601654-8263-425e-8d8f-c5bbeacbbe06
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
