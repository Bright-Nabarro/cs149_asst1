
static double ispc_dist(
		uniform double* varying x,
		uniform double* varying y,
		uniform int nDim)
{
	double accum = 0.0;

	foreach (i = 0 ... nDim)
	{
		accum += pow((x[i] - y[i]), 2);
	}
	return sqrt(accum);
}

export void ispc_compute_assign(
		uniform int start,
		uniform int end,
		uniform double data[],
		uniform double clusterCentroids[],
		uniform int clusterAssignments[],
		uniform double minDist[],
		uniform int M,
		uniform int N)
{
	foreach (m = 0 ... M)
	{
		minDist[m] = 1e30;
		clusterAssignments[m] = -1;
	}

	for(uniform int k = start; k < end; ++k)
	{
		foreach(m = 0 ... M)
		{
			
		}
	}
}
