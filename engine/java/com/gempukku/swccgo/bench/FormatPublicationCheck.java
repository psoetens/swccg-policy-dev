package com.gempukku.swccgo.bench;

import com.gempukku.swccgo.game.*;
import com.gempukku.swccgo.game.formats.DefaultSwccgFormat;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicBoolean;

/** Deterministic concurrent first-access test: park construction before its first
 * card, then make every other reader attempt to enumerate titles and card types. */
public final class FormatPublicationCheck {
    public static void main(String[] args) throws Exception {
        var entered=new CountDownLatch(1);var release=new CountDownLatch(1);
        var first=new AtomicBoolean(true);
        var format=new DefaultSwccgFormat(new SwccgCardBlueprintLibrary(),"race",false,false,false) {
            @Override public void validateCard(String bp,boolean skip) throws DeckInvalidException {
                if(first.compareAndSet(true,false)) {
                    entered.countDown();
                    try {if(!release.await(10,TimeUnit.SECONDS))throw new AssertionError("release timeout");}
                    catch(InterruptedException e){throw new AssertionError(e);}
                }
                super.validateCard(bp,skip);
            }
        };
        ExecutorService pool=Executors.newFixedThreadPool(9);
        try {
            Future<List<SwccgCardBlueprint>> builder=pool.submit(format::getAllCardBlueprintsValidInFormat);
            if(!entered.await(10,TimeUnit.SECONDS))throw new AssertionError("construction not entered");
            List<Future<List<SwccgCardBlueprint>>> readers=new ArrayList<>();
            CountDownLatch attempting=new CountDownLatch(8);
            for(int i=0;i<8;i++)readers.add(pool.submit(()-> {
                attempting.countDown();return format.getAllCardBlueprintsValidInFormat();
            }));
            if(!attempting.await(10,TimeUnit.SECONDS))throw new AssertionError("readers not started");
            // A pre-publication reader must block until construction is released.
            try {readers.get(0).get(100,TimeUnit.MILLISECONDS);throw new AssertionError("published partial list");}
            catch(TimeoutException expected) {}
            release.countDown();
            List<SwccgCardBlueprint> completed=builder.get(30,TimeUnit.SECONDS);
            if(completed.isEmpty())throw new AssertionError("empty completed list");
            for(var f:readers) {
                List<SwccgCardBlueprint> got=f.get(30,TimeUnit.SECONDS);
                if(got!=completed)throw new AssertionError("different publication");
                for(var bp:got) {bp.getTitle();bp.getCardCategory();}
            }
            try {completed.clear();throw new AssertionError("mutable publication");}
            catch(UnsupportedOperationException expected) {}
            System.out.println("immutable concurrent title/type enumeration: "+completed.size()+" cards, 9 readers");
        } finally {release.countDown();pool.shutdownNow();}
    }
}
