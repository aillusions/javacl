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

       long kernel_0 = createKernel_0(usDev, program, errcode_ret);
       long kernel_1 = createKernel_1(usDev, program, errcode_ret);
       long kernel_2 = createKernel_2(usDev, program, errcode_ret);

        {
            int errcode = clEnqueueNDRangeKernel(
                    clQueue,
                    kernel_0,
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
            int errcode = clEnqueueNDRangeKernel(
                    clQueue,
                    kernel_1,
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
            int errcode = clEnqueueNDRangeKernel(
                    clQueue,
                    kernel_2,
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

    static int patNum = 2;
    static int nrows = 2048;
    static int ncols = 2560;
    static int round = nrows * ncols;

    /**
     * 0
     */
    public static long createKernel_0(UsefulDevice usDev,
                                      LoadedProgram program,
                                      IntBuffer errcode_ret) {

        long clKernel = program.getKernel(ec_add_grid);

        {
            int sizeof = round_up_pow2(32 * 2 * round, 4096); // 335544320 ?
            long points_out = clCreateBuffer(usDev.getContext(), CL_MEM_READ_WRITE, sizeof, errcode_ret);
            clRetainMemObject(points_out);
            PointerBuffer mem_list = BufferUtils.createPointerBuffer(1);
            mem_list.put(0, points_out);
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

        {
            int sizeof = round_up_pow2(32 * 2 * ncols, 4096); // 163840 ?
            long row_in = clCreateBuffer(usDev.getContext(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeof, errcode_ret);
            clRetainMemObject(row_in);
            PointerBuffer mem_list = BufferUtils.createPointerBuffer(1);
            mem_list.put(0, row_in);
            clSetKernelArg(clKernel, 2, mem_list);
        }

        {
            int sizeof = 32 * 2 * nrows; // ?
            long col_in = clCreateBuffer(usDev.getContext(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeof, errcode_ret);
            clRetainMemObject(col_in);
            PointerBuffer mem_list = BufferUtils.createPointerBuffer(1);
            mem_list.put(0, col_in);
            clSetKernelArg(clKernel, 3, mem_list);
        }

        return  clKernel;
    }

    /**
     * 1
     */
    public static long createKernel_1(UsefulDevice usDev,
                                      LoadedProgram program,
                                      IntBuffer errcode_ret) {

        long clKernel = program.getKernel(heap_invert);

        {
            int sizeof = round_up_pow2(32 * 2 * round, 4096); // 335544320 ?
            long z_heap = clCreateBuffer(usDev.getContext(), CL_MEM_READ_WRITE, sizeof, errcode_ret);
            clRetainMemObject(z_heap);
            PointerBuffer mem_list = BufferUtils.createPointerBuffer(1);
            mem_list.put(0, z_heap);
            clSetKernelArg(clKernel, 0, mem_list);
        }

        {
            int batch = 256;
            clSetKernelArg1i(clKernel, 1, batch);
        }

        return  clKernel;
    }

    /**
     * 2
     */
    public static long createKernel_2(UsefulDevice usDev,
                                      LoadedProgram program,
                                      IntBuffer errcode_ret) {

        long clKernel = program.getKernel(hash_ec_point_search_prefix);

        {
            PointerBuffer found = BufferUtils.createPointerBuffer(1);
            clSetKernelArg(clKernel, 0, found);
        }

        {
            int sizeof = round_up_pow2(32 * 2 * round, 4096); // 335544320 ?
            long points_in = clCreateBuffer(usDev.getContext(), CL_MEM_READ_WRITE, sizeof, errcode_ret);
            clRetainMemObject(points_in);
            PointerBuffer mem_list = BufferUtils.createPointerBuffer(1);
            mem_list.put(0, points_in);
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

        {
            int sizeof = 40 * patNum;
            long target_table = clCreateBuffer(usDev.getContext(), CL_MEM_READ_WRITE, sizeof, errcode_ret);
            clRetainMemObject(target_table);
            PointerBuffer mem_list = BufferUtils.createPointerBuffer(1);
            mem_list.put(0, target_table);
            clSetKernelArg(clKernel, 3, mem_list);
        }

        clSetKernelArg1i(clKernel, 4, 1);

        return  clKernel;
    }

}

//IntBuffer buffer0 = stack.callocInt(1);
//ByteBuffer col_in = BufferUtils.createByteBuffer();
//clRetainMemObject(col_in);
//clSetKernelArg(clKernel, 3, col_in);