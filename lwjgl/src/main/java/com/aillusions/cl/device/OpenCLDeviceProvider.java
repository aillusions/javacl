package com.aillusions.cl.device;

import org.lwjgl.PointerBuffer;
import org.lwjgl.opencl.CL;
import org.lwjgl.opencl.CLCapabilities;
import org.lwjgl.system.MemoryStack;

import java.nio.IntBuffer;

import static com.aillusions.cl.demo.InfoUtil.checkCLError;
import static com.aillusions.cl.demo.InfoUtil.getDeviceInfoInt;
import static com.aillusions.cl.demo.InfoUtil.getDeviceInfoStringUTF8;
import static org.lwjgl.opencl.CL10.*;

/**
 * @author aillusions
 */
public class OpenCLDeviceProvider {

    public static UsefulDevice getUsefulDevice(MemoryStack stack, IntBuffer errcode_ret) {
        IntBuffer pi = stack.mallocInt(1);
        checkCLError(clGetPlatformIDs(null, pi));
        if (pi.get(0) == 0) {
            throw new RuntimeException("No OpenCL platforms found.");
        }

        PointerBuffer platforms = stack.mallocPointer(pi.get(0));
        checkCLError(clGetPlatformIDs(platforms, (IntBuffer) null));

        PointerBuffer ctxProps = stack.mallocPointer(3);
        ctxProps
                .put(0, CL_CONTEXT_PLATFORM)
                .put(2, 0);

        System.out.println();

        for (int p = 0; p < platforms.capacity(); p++) {
            long platform = platforms.get(p);
            ctxProps.put(1, platform);

            CLCapabilities platformCaps = CL.createPlatformCapabilities(platform);
            checkCLError(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, null, pi));

            PointerBuffer devices = stack.mallocPointer(pi.get(0));
            checkCLError(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, devices, (IntBuffer) null));


            for (int d = 0; d < devices.capacity(); d++) {
                long device = devices.get(d);

                CLCapabilities caps = CL.createDeviceCapabilities(device, platformCaps);

                String deviceName = getDeviceInfoStringUTF8(device, CL_DEVICE_NAME);

                if (!caps.OpenCL11
                        || deviceName.contains("CPU")
                        || deviceName.contains("Intel")
                        || deviceName.contains("Iris")) {
                    continue;
                }

                int addressBits = getDeviceInfoInt(device, CL_DEVICE_ADDRESS_BITS);

                // printDeviceInfo(device, deviceName + " -->> CL_DEVICE_OPENCL_C_VERSION", CL_DEVICE_OPENCL_C_VERSION);

                return new UsefulDevice(platform, device, deviceName, ctxProps, errcode_ret, addressBits);
            }
        }

        return null;
    }
}
