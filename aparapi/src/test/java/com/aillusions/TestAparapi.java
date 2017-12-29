package com.aillusions;

import com.aparapi.Kernel;
import com.aparapi.Range;
import com.aparapi.device.Device;
import com.aparapi.device.OpenCLDevice;
import junit.framework.TestCase;

import java.util.Arrays;

/**
 * http://aparapi.com/introduction/getting-started.html
 * <p>
 * -Xmx10g
 * -Dcom.aparapi.enableShowGeneratedOpenCL=true
 * –Dcom.aparapi.enableShowExecutionModes=true
 *
 * @author aillusions
 */
public class TestAparapi extends TestCase {

    static final int arraySize = Integer.MAX_VALUE / 5000;
    final int iterations = 1;

    static final float inA_gl[] = new float[arraySize];
    static final float inB_gl[] = new float[arraySize];

    static {
        Arrays.fill(inA_gl, 1);
        Arrays.fill(inB_gl, 2);

        assert (inA_gl.length == inB_gl.length);
    }

    public void testCPU() {

        long start = System.currentTimeMillis();

        final float inA[] = inA_gl;
        final float inB[] = inB_gl;
        final float[] result = new float[arraySize];

        for (int j = 0; j < iterations; j++) {
            for (int i = 0; i < result.length; i++) {
                result[i] = (float) (Math.cos(Math.sin(inA[i])) + Math.sin(Math.cos(inB[i])));
            }
        }

        System.out.println("testCPU: in " + (System.currentTimeMillis() - start) + " ms (" + result.length + " items)");
    }


    public void testGpu() {

        long start = System.currentTimeMillis();

        final float inA[] = inA_gl;
        final float inB[] = inB_gl;
        final float[] result = new float[arraySize];

        Kernel kernel = new Kernel() {
            @Override
            public void run() {
                int i = getGlobalId();
                result[i] = (float) (Math.cos(Math.sin(inA[i])) + Math.sin(Math.cos(inB[i])));
            }
        };

        Device device = Device.best();

        Range range = OpenCLDevice.bestGPU().createRange(result.length);
        // Range range = JavaDevice.THREAD_POOL.createRange(result.length);
        // Range range = Range.create(result.length);

        for (int j = 0; j < iterations; j++) {
            kernel.execute(range);
        }

        //System.out.println("Execution mode = " + kernel.getExecutionMode());

        // if (!kernel.getExecutionMode().equals(Kernel.EXECUTION_MODE.GPU)) {
        //    System.out.println("Kernel did not execute on the GPU!");
        // }

        System.out.println("testGpu: in " + (System.currentTimeMillis() - start) + " ms (" + result.length + " items)");
    }
}
