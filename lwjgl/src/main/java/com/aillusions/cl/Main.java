package com.aillusions.cl;

import com.aillusions.cl.device.OpenCLDeviceProvider;
import com.aillusions.cl.device.UsefulDevice;
import com.aillusions.cl.programm.LoadedProgram;
import com.aillusions.cl.programm.ProgramLoader;
import org.lwjgl.BufferUtils;
import org.lwjgl.PointerBuffer;
import org.lwjgl.opencl.CL10;
import org.lwjgl.system.MemoryStack;

import java.io.IOException;
import java.nio.IntBuffer;

import static com.aillusions.cl.demo.InfoUtil.checkCLError;
import static com.aillusions.cl.programm.ProgramLoader.getFileNamePath;
import static org.lwjgl.opencl.CL10.*;
import static org.lwjgl.system.MemoryStack.stackPush;

/**
 * @author aillusions
 */
public class Main {

    private static final String ec_add_grid = "ec_add_grid"; // [0]
    private static final String heap_invert = "heap_invert"; // [1]
    private static final String hash_ec_point_search_prefix = "hash_ec_point_search_prefix"; // [2]

    private static final String calc_addr_file = getFileNamePath("calc_addrs");

    private static final int n = 10;

    private static final float anArray[] = new float[n];

    static {
        for (int i = 0; i < n; i++) {
            anArray[i] = i;
        }
    }

    public static void main(String... arg) throws IOException {

        MemoryStack stack = stackPush();
        IntBuffer errcode_ret = stack.callocInt(1);

        UsefulDevice usDev = OpenCLDeviceProvider.getUsefulDevice(stack, errcode_ret);
        if (usDev == null) {
            System.out.println("No useful device found.");
            return;
        } else {
            System.out.println("Useful device: " + usDev.toString());
        }

        LoadedProgram program = ProgramLoader.loadProgramm(
                errcode_ret,
                usDev,
                calc_addr_file,
                ec_add_grid,
                heap_invert,
                // "hash_ec_point_get", // ??
                hash_ec_point_search_prefix);


        PointerBuffer event = BufferUtils.createPointerBuffer(1);

        long clQueue = clCreateCommandQueue(usDev.getContext(), usDev.getDevice(), 0, errcode_ret);

        PointerBuffer globalws = BufferUtils.createPointerBuffer(1);
        globalws.put(0, 10);

        PointerBuffer invws = BufferUtils.createPointerBuffer(1);
        invws.put(0, 10);

        {
            long clKernel = program.getKernel(ec_add_grid);

            long bufferArg1 = SumClCalc.allocateBufferFor(errcode_ret, usDev, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, anArray);

            clSetKernelArg1p(clKernel, 0, bufferArg1);
            clSetKernelArg1p(clKernel, 1, bufferArg1);
            clSetKernelArg1p(clKernel, 2, bufferArg1);
            clSetKernelArg1p(clKernel, 3, bufferArg1);

            int errcode = clEnqueueNDRangeKernel(
                    clQueue,
                    clKernel,
                    1,
                    null,
                    globalws,
                    null,
                    null,
                    event);
            checkCLError(errcode);

            CL10.clWaitForEvents(event);
        }

        {
            long clKernel = program.getKernel(heap_invert);

            long bufferArg1 = SumClCalc.allocateBufferFor(errcode_ret, usDev, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, anArray);

            clSetKernelArg1p(clKernel, 0, bufferArg1);

            clSetKernelArg1i(clKernel, 1, 256);

            int errcode = clEnqueueNDRangeKernel(
                    clQueue,
                    clKernel,
                    1,
                    null,
                    invws,
                    null,
                    null,
                    event);
            checkCLError(errcode);

            CL10.clWaitForEvents(event);
        }

        {
            long clKernel = program.getKernel(hash_ec_point_search_prefix);

            long bufferArg1 = SumClCalc.allocateBufferFor(errcode_ret, usDev, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, anArray);

            clSetKernelArg1p(clKernel, 0, bufferArg1);
            clSetKernelArg1p(clKernel, 1, bufferArg1);
            clSetKernelArg1p(clKernel, 2, bufferArg1);
            clSetKernelArg1p(clKernel, 3, bufferArg1);

            clSetKernelArg1i(clKernel, 4, 1);

            int errcode = clEnqueueNDRangeKernel(
                    clQueue,
                    clKernel,
                    1,
                    null,
                    globalws,
                    null,
                    null,
                    event);
            checkCLError(errcode);

            CL10.clWaitForEvents(event);
        }

        program.clear();
        usDev.clear();
    }
}
