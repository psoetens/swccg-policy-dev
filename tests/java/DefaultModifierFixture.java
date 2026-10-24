package com.gempukku.swccgo.logic.modifiers;

/** Real reflected modifier package, also recognized by the frozen schema-3 walker. */
public final class DefaultModifierFixture extends AbstractModifier {
    private final int storedAmount;
    private final boolean disabled = false;

    public DefaultModifierFixture(int amount) {
        super(null, null, null, null);
        storedAmount = amount;
    }
}
