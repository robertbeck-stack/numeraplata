package app.numby.alpr

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class PlateTest {
  @Test
  fun validAndInvalidPlates() {
    assertEquals("AB123", extractPlate("AB-123"))
    assertEquals("ABC99", extractPlate(" ab c99 "))
    assertEquals("LOL46", extractPlate("IS LO 25 L46"))
    assertEquals("LOL46", extractPlate("L0 25 L46"))
    assertEquals("AB120", extractPlate("AB 12O"))
    assertNull(extractPlate("PARIS"))
    assertNull(extractPlate("S069"))
    assertNull(extractPlate("4SBEKO"))
    assertNull(extractPlate("WINTPSP2"))
    assertTrue(isSameCar("VSD07", "VSO07"))
    assertTrue(isSameCar("VSD07", "VSE07"))
    assertTrue(isSameCar("XLS53", "IXL53"))
    assertFalse(isSameCar("VSD07", "ABC12"))
  }
}
