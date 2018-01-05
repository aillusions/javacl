package com.aillusions.cl.programm;

import lombok.AllArgsConstructor;
import lombok.Data;
import org.lwjgl.opencl.CL10;

/**
 * @author aillusions
 */
@Data
@AllArgsConstructor
public class LoadedProgram {

    private long clProgram;
    private long clKernel;

    public void clear() {
        // Destroy our kernel and program
        CL10.clReleaseKernel(clKernel);
        CL10.clReleaseProgram(clProgram);
    }
}
