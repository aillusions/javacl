package com.aillusions.cl.programm;

import lombok.AllArgsConstructor;
import lombok.Data;
import org.lwjgl.opencl.CL10;

import java.util.HashMap;
import java.util.Map;

/**
 * @author aillusions
 */
@Data
@AllArgsConstructor
public class LoadedProgram {

    private long clProgram;
    private Map<String, Long> kernels = new HashMap<>();

    public long getKernel(String progName){
        return kernels.get(progName).longValue();
    }

    public void clear() {
        kernels.values().forEach(CL10::clReleaseKernel);
        CL10.clReleaseProgram(clProgram);
    }
}
