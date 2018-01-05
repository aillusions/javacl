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


        long clQueue = clCreateCommandQueue(usDev.getContext(), usDev.getDevice(), 0, errcode_ret);

        PointerBuffer globalws = BufferUtils.createPointerBuffer(1);
        globalws.put(0, 10);

        PointerBuffer invws = BufferUtils.createPointerBuffer(1);
        invws.put(0, 10);

        {
            long clKernel = program.getKernel(ec_add_grid);

            PointerBuffer ev = BufferUtils.createPointerBuffer(1);

            int errcode = clEnqueueNDRangeKernel(
                    clQueue,
                    clKernel,
                    1,
                    null,
                    globalws,
                    null,
                    null,
                    null);
            checkCLError(errcode);

            CL10.clWaitForEvents(ev);
        }

        {
            long clKernel = program.getKernel(heap_invert);

            clSetKernelArg1i(clKernel, 1, 256);

            int errcode = clEnqueueNDRangeKernel(
                    clQueue,
                    clKernel,
                    1,
                    null,
                    invws,
                    null,
                    null,
                    null);
            checkCLError(errcode);
        }

        {
            long clKernel = program.getKernel(hash_ec_point_search_prefix);

            int errcode = clEnqueueNDRangeKernel(
                    clQueue,
                    clKernel,
                    1,
                    null,
                    globalws,
                    null,
                    null,
                    null);
            checkCLError(errcode);
        }

        program.clear();
        usDev.clear();
    }
}
