package com.aillusions;

import com.aparapi.Kernel;
import com.aparapi.Range;
import junit.framework.TestCase;

import java.util.Arrays;

/**
 * -Xmx10g
 *
 * @author aillusions
 */
public class TestAparapi extends TestCase {

    static final int arraySize = Integer.MAX_VALUE / 50;
    final int iterations = 50;

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
                result[i] = inA[i] + inB[i];
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
                result[i] = inA[i] + inB[i];
            }
        };

        Range range = Range.create(result.length);
        for (int j = 0; j < iterations; j++) {
            kernel.execute(range);
        }

        System.out.println("testGpu: in " + (System.currentTimeMillis() - start) + " ms (" + result.length + " items)");
    }
}
