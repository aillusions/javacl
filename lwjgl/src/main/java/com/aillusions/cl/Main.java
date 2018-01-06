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

    /*public static boolean is_pow2(int v) {
        return !(v & (v - 1));
    }*/

    public static int round_up_pow2(int x, int a) {
        return (((x) + ((a) - 1)) & ~((a) - 1));
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

        createKernel_0(usDev, program, errcode_ret, clQueue, globalws, invws, event);
        createKernel_1(usDev, program, errcode_ret, clQueue, globalws, invws, event);
        createKernel_2(usDev, program, errcode_ret, clQueue, globalws, invws, event);

        program.clear();
        usDev.clear();
    }

    static int nrows = 2048;
    static int ncols = 2560;
    static int round = nrows * ncols;

    public static void createKernel_0(UsefulDevice usDev,
                                      LoadedProgram program,
                                      IntBuffer errcode_ret,
                                      long clQueue,
                                      PointerBuffer globalws,
                                      PointerBuffer invws,
                                      PointerBuffer event) {

        long clKernel = program.getKernel(ec_add_grid);

        long bufferArg1 = SumClCalc.allocateBufferFor(errcode_ret, usDev, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, anArray);

        {
            int sizeof = round_up_pow2(32 * 2 * round, 4096); // 335544320 ?
            long z_heap = clCreateBuffer(usDev.getContext(), CL_MEM_READ_WRITE, sizeof, errcode_ret);
            clRetainMemObject(z_heap);
            PointerBuffer mem_list = BufferUtils.createPointerBuffer(1);
            mem_list.put(0, z_heap);
            clSetKernelArg(clKernel, 0, mem_list);
        }

        {
            int sizeof = round_up_pow2(32 * 2 * round, 4096); // 335544320 ?
            long z_heap = clCreateBuffer(usDev.getContext(), CL_MEM_READ_WRITE, sizeof, errcode_ret);
            clRetainMemObject(z_heap);
            PointerBuffer mem_list = BufferUtils.createPointerBuffer(1);
            mem_list.put(0, z_heap);
            clSetKernelArg(clKernel, 1, mem_list);
        }

        clSetKernelArg1p(clKernel, 2, bufferArg1);

        {
            int sizeof = 32 * 2 * nrows; // ?
            long col_in = clCreateBuffer(usDev.getContext(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeof, errcode_ret);
            clRetainMemObject(col_in);
            PointerBuffer mem_list = BufferUtils.createPointerBuffer(1);
            mem_list.put(0, col_in);
            clSetKernelArg(clKernel, 3, mem_list);
        }

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

    public static void createKernel_1(UsefulDevice usDev,
                                      LoadedProgram program,
                                      IntBuffer errcode_ret,
                                      long clQueue,
                                      PointerBuffer globalws,
                                      PointerBuffer invws,
                                      PointerBuffer event) {

        long clKernel = program.getKernel(heap_invert);

        {
            int sizeof = round_up_pow2(32 * 2 * round, 4096); // 335544320 ?
            long z_heap = clCreateBuffer(usDev.getContext(), CL_MEM_READ_WRITE, sizeof, errcode_ret);
            clRetainMemObject(z_heap);
            PointerBuffer mem_list = BufferUtils.createPointerBuffer(1);
            mem_list.put(0, z_heap);
            clSetKernelArg(clKernel, 0, mem_list);
        }

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

    public static void createKernel_2(UsefulDevice usDev,
                                      LoadedProgram program,
                                      IntBuffer errcode_ret,
                                      long clQueue,
                                      PointerBuffer globalws,
                                      PointerBuffer invws,
                                      PointerBuffer event) {

        long clKernel = program.getKernel(hash_ec_point_search_prefix);


        //IntBuffer buffer0 = stack.callocInt(1);

        {
            PointerBuffer found = BufferUtils.createPointerBuffer(1);
            clSetKernelArg(clKernel, 0, found);
        }

        long bufferArg1 = SumClCalc.allocateBufferFor(errcode_ret, usDev, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, anArray);

        {
            int sizeof = round_up_pow2(32 * 2 * round, 4096); // 335544320 ?
            long z_heap = clCreateBuffer(usDev.getContext(), CL_MEM_READ_WRITE, sizeof, errcode_ret);
            clRetainMemObject(z_heap);
            PointerBuffer mem_list = BufferUtils.createPointerBuffer(1);
            mem_list.put(0, z_heap);
            clSetKernelArg(clKernel, 1, mem_list);
        }

        {
            int sizeof = round_up_pow2(32 * 2 * round, 4096); // 335544320 ?
            long z_heap = clCreateBuffer(usDev.getContext(), CL_MEM_READ_WRITE, sizeof, errcode_ret);
            clRetainMemObject(z_heap);
            PointerBuffer mem_list = BufferUtils.createPointerBuffer(1);
            mem_list.put(0, z_heap);
            clSetKernelArg(clKernel, 2, mem_list);
        }

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

}


//ByteBuffer col_in = BufferUtils.createByteBuffer();
//clRetainMemObject(col_in);
//clSetKernelArg(clKernel, 3, col_in);