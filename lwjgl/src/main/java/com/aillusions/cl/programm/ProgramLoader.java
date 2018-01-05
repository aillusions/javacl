package com.aillusions.cl.programm;

import com.aillusions.cl.device.UsefulDevice;
import org.lwjgl.BufferUtils;
import org.lwjgl.PointerBuffer;
import org.lwjgl.opencl.CLProgramCallback;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.IntBuffer;
import java.util.concurrent.CountDownLatch;

import static com.aillusions.cl.demo.IOUtil.ioResourceToByteBuffer;
import static com.aillusions.cl.demo.InfoUtil.*;
import static org.lwjgl.opencl.CL10.*;
import static org.lwjgl.system.MemoryUtil.NULL;

/**
 * @author aillusions
 */
public class ProgramLoader {

    public static LoadedProgram loadProgramm(IntBuffer errcode_ret, UsefulDevice usDev, String progName) throws IOException {

        long device = usDev.getDevice();
        long context = usDev.getContext();

        PointerBuffer strings = BufferUtils.createPointerBuffer(1);
        PointerBuffer lengths = BufferUtils.createPointerBuffer(1);

        ByteBuffer source = ioResourceToByteBuffer(getFileNamePath(progName), 4096);
        strings.put(0, source);
        lengths.put(0, source.remaining());

        long clProgram = clCreateProgramWithSource(context, strings, lengths, errcode_ret);
        checkCLError(errcode_ret);

        StringBuilder options = new StringBuilder("");

        CountDownLatch latch = new CountDownLatch(1);

        CLProgramCallback buildCallback;
        int errcode = clBuildProgram(clProgram, device, options, buildCallback = CLProgramCallback.create((program, user_data) -> {
            System.out.println(String.format(
                    "The cl_program [0x%X] was built %s",
                    program,
                    getProgramBuildInfoInt(program, device, CL_PROGRAM_BUILD_STATUS) == CL_SUCCESS ? "successfully" : "unsuccessfully"
            ));
            String log = getProgramBuildInfoStringASCII(program, device, CL_PROGRAM_BUILD_LOG);
            if (!log.isEmpty()) {
                System.out.println(String.format("BUILD LOG:\n----\n%s\n-----", log));
            }

            latch.countDown();
        }), NULL);
        checkCLError(errcode);

        try {
            latch.await();
        } catch (InterruptedException e) {
            throw new RuntimeException(e);
        }

        buildCallback.free();

        // init kernel with constants
        long clKernel = clCreateKernel(clProgram, progName, errcode_ret);
        checkCLError(errcode_ret);


        return new LoadedProgram(clProgram, clKernel);
    }

    private static String getFileNamePath(String progName) {
        return progName + ".cl";
    }
}
