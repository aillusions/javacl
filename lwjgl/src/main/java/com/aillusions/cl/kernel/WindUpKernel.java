package com.aillusions.cl.kernel;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.ToString;

/**
 * @author aillusions
 */
@Data
@AllArgsConstructor
@ToString
public class WindUpKernel {
    private long kernel;
    private long[] buffers;
}
