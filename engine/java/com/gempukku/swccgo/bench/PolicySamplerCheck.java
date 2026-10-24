package com.gempukku.swccgo.bench;
import com.google.gson.Gson;
import java.util.*;

/** Sampler parity fixtures include the rare multi-pick and exclusion paths. */
public final class PolicySamplerCheck {
    public static void main(String[] args) {
        Gson json=new Gson();
        float[] logits={-1.25f,0.75f,2.0f,-3.0f,0.125f};
        boolean[] none={true,false,false,false,false};
        for(double temperature:new double[]{0.7,1.0,1.5}) for(int k:new int[]{1,3,8}) for(int seed=0;seed<32;seed++) {
            int[] excluded=seed%2==0?new int[]{1,3}:new int[0];
            var s=PolicySampler.sample(logits,logits.length,none,k,temperature,excluded,new SplittableRandom(seed));
            if(new HashSet<>(s.seq).size()!=s.seq.size())throw new AssertionError("replacement");
            if(s.seq.size()>1 && s.seq.contains(0))throw new AssertionError("pass mixed with picks");
            for(int x:excluded)if(s.seq.contains(x))throw new AssertionError("excluded selection");
            System.out.println(json.toJson(Map.of("logits",logits,"none",none,"excluded",excluded,
                "temperature",temperature,"seq",s.seq,"logp",s.logp)));
        }
        float[] extreme={-1000,0};boolean[] noPass={false,false};
        double logp=PolicySampler.logProb(extreme,2,noPass,1,1,null,List.of(0));
        if(Math.abs(logp+1000)>1e-6)throw new AssertionError("behavior probability was clipped");
        System.out.println(json.toJson(Map.of("logits",extreme,"none",noPass,"excluded",new int[0],
            "temperature",1,"seq",List.of(0),"logp",logp)));
    }
}
