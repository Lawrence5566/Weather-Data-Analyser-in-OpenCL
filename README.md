# Weather-Data-Analyser-in-OpenCL
Weather Data Analyser in OpenCL using C++ for university module

Given a large dataset (1,873,106 values) of temperatures, without using any external libraries (e.g. Boost), this project uses OpenCL parallel computing to find:

-Mean
-Min
-Max
-Standard deviation
-1st Quartile
-3rd Quartile
-Sorting (unfinished)

The program uses common parallel patterns such as map, reduction and scan.
The program reports memory transfer, kernel execution and total program execution times for performance assessment.
