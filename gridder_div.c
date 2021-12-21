/*
 * Parallel density gridder
 *
 * Takes 3D particle data and assigns to a uniform density grid.
 *
 * This version also calculates the divergence along each co-ordinate axis
 *
 */


#include "hdf5.h"
#include <mpi.h>
#include <gmp.h>

#include <stdlib.h>
#include <dirent.h>
#include <string.h>
#include <math.h>

#include <stdlib.h>
#include "read_config.h"

// TODO: set as runtime value
// PartType1: Dark Matter
#define DATASETNAME_POS "/PartType1/Coordinates"  // dataset within hdf5 file 
#define DATASETNAME_VEL "/PartType1/Velocity"  // dataset within hdf5 file 


// function initialisers
int count_files(const char *, const char *);
const char *get_filename_ext(const char *);
int offset(int, int, int, int);
//void NGP(int *, int, float, float, float);
void NGP(int *,                        // sum of number
	 float *, float *, float *,    // sum of pos
	 float *, float *, float *,    // sum of vel
	 float *, float *, float *,    // sum of pos * pos
	 float *, float *, float *,    // sum of pos * vel
	 int,                          // size of grid along each axis
	 float, float, float,          // input grid position
	 float, float, float,          // input position
	 float, float, float);         // input velocity

int main (int argc, char **argv) {

    int i,j,k;

    /*
     * Read config file
     */
    struct config_struct config;

    // read_config_file("config.txt", &config);
    read_config_file(argv[1], &config);

    char * input_directory = config.input_dir; 
    char * output_file_N = config.output_file_N;
    char * output_file_div = config.output_file_div;

    /*
     * Initialise weight grid
     */
    int grid_dims = config.grid_dims; //grid_size + 1; 
    double sim_dims = config.sim_dims;  // simulation dimensions
    
    hid_t 	file, dataset_pos, dataset_vel, dataspace;   // handles
    herr_t 	status;
    int		status_n;
    hsize_t 	dims[2];           			// dataset dimensions
	
    /*
     * Initialize MPI
     */
    int mpi_size, mpi_rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    
    int ierr;  // store error values for MPI operations
    int root_process = 0;  // set root process to zero processor  

    if(mpi_rank == 0){
        printf("\nInput directory: %s",input_directory);
	printf("\nOutput file for density: %s", output_file_N);
	printf("\nOutput file for divergences: %s", output_file_div);
        printf("\nGrid dimensions: %d", grid_dims);
        printf("\nSimulation dimensions: %lf", sim_dims);
    }
    /*
     * Count hdf5 files in specified directory
     */		
    const char * extension = "hdf5";
    int file_count = count_files(input_directory,extension);
    
    /*
     * Find hdf5 files in specified directory
     */
    char **files = malloc(sizeof(char*) * file_count);
    int ticker = 0;
    DIR * dirp;
    struct dirent * entry;
    dirp = opendir(input_directory);
    while ((entry = readdir(dirp)) != NULL) {
	if (entry->d_type == DT_REG && !strcmp(get_filename_ext((const char *)entry->d_name), "hdf5")) {  // If the entry is a regular file, with hdf5 extension..
	    files[ticker] = calloc(sizeof(char*),sizeof(char *));  // allocate space in files array for this string
	    strcpy(files[ticker], entry->d_name);  // ...store the filename
	    strcat(files[ticker],"\0");  // add string end character to convert character array to string
	    ticker++;
	}
    }
    closedir(dirp);

    /*
     *  find number of files for given processor
     */
    int proc_files = file_count / mpi_size;
    if(mpi_rank < fmod(file_count,mpi_size)) proc_files++;

    /*
     * A 3D array is too big for native initialisation, and a pain using malloc.
     * So, create a 1D array and use a custom offset function (see end)
     */
    long int grid_size = pow(grid_dims, 3); 	
    char * fullname;
    long long int particle_count = 0;
    long long int particle_count_slave = 0;
    int *N = calloc(grid_size, sizeof *N);
    float *X = calloc(grid_size, sizeof *X);
    float *Y = calloc(grid_size, sizeof *Y);
    float *Z = calloc(grid_size, sizeof *Z);
    float *Vx = calloc(grid_size, sizeof *Vx);
    float *Vy = calloc(grid_size, sizeof *Vy);
    float *Vz = calloc(grid_size, sizeof *Vz);
    float *XX = calloc(grid_size, sizeof *XX);
    float *YY = calloc(grid_size, sizeof *YY);
    float *ZZ = calloc(grid_size, sizeof *ZZ);
    float *XVx = calloc(grid_size, sizeof *XVx);
    float *YVy = calloc(grid_size, sizeof *YVy);
    float *ZVz = calloc(grid_size, sizeof *ZVz);
    int *N_slave = calloc(grid_size, sizeof *N);
    float *X_slave = calloc(grid_size, sizeof *X);
    float *Y_slave = calloc(grid_size, sizeof *Y);
    float *Z_slave = calloc(grid_size, sizeof *Z);
    float *Vx_slave = calloc(grid_size, sizeof *Vx);
    float *Vy_slave = calloc(grid_size, sizeof *Vy);
    float *Vz_slave = calloc(grid_size, sizeof *Vz);
    float *XX_slave = calloc(grid_size, sizeof *XX);
    float *YY_slave = calloc(grid_size, sizeof *YY);
    float *ZZ_slave = calloc(grid_size, sizeof *ZZ);
    float *XVx_slave = calloc(grid_size, sizeof *XVx);
    float *YVy_slave = calloc(grid_size, sizeof *YVy);
    float *ZVz_slave = calloc(grid_size, sizeof *ZVz);

    for(i = mpi_rank; i<file_count; i+=mpi_size){
	printf("%d %s\n",i,files[i]);
	fullname = malloc(sizeof(char) * (strlen(input_directory) + strlen(files[i]) + 1));  // allocate space for concatenated full name and location
	*fullname = '\0';
	strcat(fullname, input_directory);  // concatenate directory and filename strings
	strcat(fullname, files[i]);
	//  Open the hdf5 file and dataset
	file = H5Fopen(fullname, H5F_ACC_RDONLY, H5P_DEFAULT);
	dataset_pos = H5Dopen(file, DATASETNAME_POS, H5P_DEFAULT);
	dataset_vel = H5Dopen(file, DATASETNAME_VEL, H5P_DEFAULT);
	free(fullname);
	dataspace = H5Dget_space(dataset_pos);    // dataspace handle
	status_n  = H5Sget_simple_extent_dims(dataspace, dims, NULL);  // get dataspace dimensions
	/*
	 *  Initialise data buffer
	 */
	int rows = dims[0];  // Number of particles?
	int cols = dims[1];  // Number of dimensions (=3)?
	particle_count_slave += rows;
	printf("%d: %d particles, %lld total\n", i, rows, particle_count_slave);
	float **data_pos; 
	float **data_vel; 
	/* 
	 * Allocate memory for new float array[row][col] 
	 */
	/* First allocate the memory for the top-level array (rows).
	   Make sure you use the sizeof a *pointer* to your data type. */
	data_pos = (float**) calloc(rows, sizeof(float*));  // Done this roundabout way in case not 3 dimensional
	data_vel = (float**) calloc(rows, sizeof(float*));
	/* Allocate a contiguous chunk of memory for the array data values.
	   Use the sizeof the data type. */
	data_pos[0] = (float*) calloc(cols*rows, sizeof(float));
	data_vel[0] = (float*) calloc(cols*rows, sizeof(float));
	/* Set the pointers in the top-level (row) array to the
	   correct memory locations in the data value chunk. */
	for (j=1; j < rows; j++) data_pos[j] = data_pos[0] + j*cols;
	for (j=1; j < rows; j++) data_vel[j] = data_vel[0] + j*cols;
	/*
	 * Read dataset back.
	 */
	status = H5Dread(dataset_pos, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &data_pos[0][0]);
	status = H5Dread(dataset_vel, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &data_vel[0][0]);
	/*
	 * Close datasets, dataspace and file
	 */
	H5Dclose(dataset_pos);
	H5Dclose(dataset_vel);
	H5Sclose(dataspace);
	H5Fclose(file);
	/*
	 * Assign to grid.
         * This version accumulates quantities needed to calculate the divergence
	 */
	float xgrid, ygrid, zgrid;
	double ratio = grid_dims / sim_dims;  // ratio of grid to simulation dimensions
	for(j = 0; j < rows; j++){  // loop through data rows 
	    xgrid = data_pos[j][0] * ratio; // x grid position
	    ygrid = data_pos[j][1] * ratio; // y grid position
	    zgrid = data_pos[j][2] * ratio; // z grid position
	    NGP(N_slave,                            // sum of number
		X_slave, Y_slave, Z_slave,          // sum of pos
		Vx_slave, Vy_slave, Vz_slave,       // sum of vel
		XX_slave, YY_slave, ZZ_slave,       // sum of pos * pos
		XVx_slave, YVy_slave, ZVz_slave,    // sum of pos * vel
		grid_dims,                          // size of grid along each axis
		xgrid, ygrid, zgrid,                // position in grid coordinates
		data_pos[j][0], data_pos[j][1], data_pos[j][2],  // input position
		data_vel[j][0], data_vel[j][1], data_vel[j][2]);  // input velocity
		}
	printf("%d complete\n", i);
	free(data_pos[0]);
	free(data_pos);
	free(data_vel[0]);
	free(data_vel);
	}
    /* Accumulate sums over all ranks */
    MPI_Reduce(&particle_count_slave, &particle_count, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(N_slave, N, grid_size, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(X_slave, X, grid_size, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(Y_slave, Y, grid_size, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(Z_slave, Z, grid_size, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(Vx_slave, Vx, grid_size, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(Vy_slave, Vy, grid_size, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(Vz_slave, Vz, grid_size, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(XX_slave, XX, grid_size, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(YY_slave, YY, grid_size, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(ZZ_slave, ZZ, grid_size, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(XVx_slave, XVx, grid_size, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(YVy_slave, YVy, grid_size, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(ZVz_slave, ZVz, grid_size, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    /*
     * Now on the master node, we can calculate our statistics and write out
     */
    if(mpi_rank == 0){
	float* divx = calloc(grid_size, sizeof *divx);
	float* divy = calloc(grid_size, sizeof *divy);
	float* divz = calloc(grid_size, sizeof *divz);
	printf("Total particles: %lld\n", particle_count);
	/* Calculate divergences: note that these need correcting for Hubble expansion */
	for (i=0; i<grid_size; i++){
	    divx[i] = (N[i]*XVx[i]-X[i]*Vx[i])/(N[i]*XX[i]-X[i]*X[i]);
	    divy[i] = (N[i]*YVy[i]-Y[i]*Vy[i])/(N[i]*YY[i]-Y[i]*Y[i]);
	    divz[i] = (N[i]*ZVz[i]-Z[i]*Vz[i])/(N[i]*ZZ[i]-Z[i]*Z[i]);
	}
	/*
	 * Write to output files
	 */
	FILE *fp;		
	fp = fopen(output_file_N, "wb");
	if (fp == NULL) printf("%s could not be opened.\n",output_file_N);		
	// Save final weight array
	fwrite(N, sizeof(int), grid_size, fp);
	fclose(fp);
	fp = fopen(output_file_div, "wb");
	if (fp == NULL) printf("%s could not be opened.\n",output_file_div);		
	/* 
	 * Save final divergence arrays
	 * Slower at this point to change to 3xN rather than Nx3, but makes more sense to co-locate
	 */
	for(i = 0; i < grid_size; i++){
	    fwrite(&divx[i], sizeof(float), 1, fp);
	    fwrite(&divy[i], sizeof(float), 1, fp);
	    fwrite(&divz[i], sizeof(float), 1, fp);
	}
	fclose(fp);
    }
    // Let's be good and free up all the memory, even though the end of the program. 
    free(N_slave);
    free(X_slave);
    free(Y_slave);
    free(Z_slave);
    free(Vx_slave);
    free(Vy_slave);
    free(Vz_slave);
    free(XX_slave);
    free(YY_slave);
    free(ZZ_slave);
    free(XVx_slave);
    free(YVy_slave);
    free(ZVz_slave);
    free(N);
    free(X);
    free(Y);
    free(Z);
    free(Vx);
    free(Vy);
    free(Vz);
    free(XX);
    free(YY);
    free(ZZ);
    free(XVx);
    free(YVy);
    free(ZVz);
    free(files[0]);
    free(files);
    ierr = MPI_Finalize();
    return 0;
}


/*
 * Nearest Grid Point assignment
 */
void NGP(int   *N,                              // sum of number
	 float *X, float *Y, float *Z,          // sum of pos
	 float *Vx, float *Vy, float *Vz,       // sum of vel
	 float *XX, float *YY, float *ZZ,       // sum of pos * pos
	 float *XVx, float *YVy, float *ZVz,    // sum of pos * vel
	 int dims,                              // size of grid along each axis
	 float xgrid, float ygrid, float zgrid, // input grid position
	 float x, float y, float z,             // input position
	 float vx, float vy, float vz){         // input velocity
    int loc;
    loc = offset((int) xgrid, (int) ygrid, (int) zgrid, dims);   // location in 1-d array
    N[loc] += 1;
    X[loc] += x;
    Y[loc] += y;
    Z[loc] += z;
    Vx[loc] += vx;
    Vy[loc] += vy;
    Vz[loc] += vz;
    XX[loc] += x*x;
    YY[loc] += y*y;
    ZZ[loc] += z*z;
    XVx[loc] += x*vx;
    YVy[loc] += y*vy;
    ZVz[loc] += z*vz;
}

/*
 * Cloud In Cell assignment
 */
void CIC(int * array, int dims, int x, int y, int z){
    
    array[offset((int)x, (int)y, (int)z, dims)] += (1 - fmod(x, 1.)) * (1 - fmod(y, 1.)) * (1 - fmod(z, 1.));
    array[offset((int)x + 1, (int)y, (int)z, dims)] += fmod(x, 1.) * (1 - fmod(y, 1.)) * (1 - fmod(z, 1.));
    
    array[offset((int)x, (int)y + 1, (int)z, dims)] += (1 - fmod(x, 1.)) * fmod(y, 1.) * (1 - fmod(z, 1.));
    array[offset((int)x + 1, (int)y + 1, (int)z, dims)] += fmod(x, 1.) * fmod(y, 1.) * (1 - fmod(z, 1.));
    
    array[offset((int)x, (int)y, (int)z + 1, dims)] += (1 - fmod(x, 1.)) * (1 - fmod(y, 1.)) * fmod(z, 1.);
    array[offset((int)x + 1, (int)y, (int)z + 1, dims)] += fmod(x, 1.) * (1 - fmod(y, 1.)) * fmod(z, 1.);
     
    array[offset((int)x, (int)y + 1, (int)z + 1, dims)] += (1 - fmod(x, 1.)) * fmod(y, 1.) * fmod(z, 1.);
    array[offset((int)x + 1, (int)y + 1, (int)z + 1, dims)] += fmod(x, 1.) * fmod(y, 1.) * fmod(z, 1.);  
 
}

/*
 * Triangular Shaped Cloud assignment
 */
void TSC(int * array, int dims, int x, int y, int z){
}


int count_files(const char * directory, const char * extension){
	/*
	 * given a directory, count the number of files in it
	 */

	int file_count = 0;
	DIR * dirp = opendir(directory);
	struct dirent * entry;

	while ((entry = readdir(dirp)) != NULL) {
		if (entry->d_type == DT_REG && !strcmp(get_filename_ext((const char *)entry->d_name), extension)) file_count++;  // If the entry is a regular file..
	}
	closedir(dirp);

	return file_count;
}


const char *get_filename_ext(const char * filename) {
	/*
	 * Given a filename, return the extension
	 */

    const char *dot = strrchr(filename, '.');
    if(!dot || dot == filename) return "";
    return dot + 1;
}


/*
 * Given a 3D array (grid_dims^3) flattened to 1D, return offset for given 3D coordinates in flat array
 */
int offset(int x, int y, int z, int grid_dims) { return ( z * grid_dims * grid_dims ) + ( y * grid_dims ) + x ; }
