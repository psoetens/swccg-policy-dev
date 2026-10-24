package com.gempukku.swccgo.bench;

import java.nio.ByteBuffer;
import java.nio.FloatBuffer;

/** Grid dimensions and owned-table checks for ABI 4; lifecycle uses NativePolicy. */
public final class NativePolicyV4 {
    private NativePolicyV4() {}
    public static native int checkTable(long slot, String tableSha);
    /** The loaded module's tensor ABI (4, or 5 = eight floats per token; plan objective-dojo Amendment 2). */
    public static native int tensorAbi(long slot);
    public static native int forward(long slot, ByteBuffer in, int b, int t, int o, int k, FloatBuffer out);
    public static native int launch(long slot, ByteBuffer in, int b, int t, int o, int k);
}
