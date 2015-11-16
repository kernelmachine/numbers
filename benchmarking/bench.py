import numpy
import timeit



if __name__ == "__main__":
    rNumpy = []
    for i in range(5,250,5):
        m1 = numpy.random.rand(i,i).astype(numpy.float32)
        m2 = numpy.random.rand(i,i).astype(numpy.float32)
        tNumpy = timeit.Timer("numpy.linalg.svd(m1)", setup = "import numpy; from __main__ import m1")
        bench = (i, numpy.mean(tNumpy.repeat(5, 1)))
        rNumpy.append(bench)

    print rNumpy
