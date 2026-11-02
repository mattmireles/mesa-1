/*
 * Copyright 2022 Alyssa Rosenzweig
 * SPDX-License-Identifier: MIT
 */

#include "agx_builder.h"
#include "agx_compiler.h"
#include "agx_debug.h"
#include "util/u_debug.h"

#define AGX_MAX_PENDING (8)

/*
 * Returns whether an instruction is asynchronous and needs a scoreboard slot
 */
static bool
instr_is_async(agx_instr *I)
{
   return agx_opcodes_info[I->op].immediates & AGX_IMMEDIATE_SCOREBOARD;
}

struct slot {
   /* Set of registers this slot is currently writing */
   BITSET_DECLARE(writes, AGX_NUM_REGS);

   /* Number of pending messages on this slot. Must not exceed
    * AGX_MAX_PENDING for correct results.
    */
   uint8_t nr_pending;

   /* Global mode: a store is pending on this slot */
   bool store;
};

/* An asynchronous instruction that writes memory and no register */
static bool
instr_is_store(agx_instr *I)
{
   if (!instr_is_async(I) || I->op == AGX_OPCODE_ATOMIC)
      return false;

   agx_foreach_dest(I, d) {
      if (!agx_is_null(I->dest[d]))
         return false;
   }

   return true;
}

/*
 * Insert waits within a block to stall after every async instruction. Useful
 * for debugging.
 */
static void
agx_insert_waits_trivial(agx_context *ctx, agx_block *block)
{
   agx_foreach_instr_in_block_safe(block, I) {
      if (instr_is_async(I)) {
         agx_builder b = agx_init_builder(ctx, agx_after_instr(I));
         agx_wait(&b, I->scoreboard);
      }
   }
}

/*
 * Scoreboard state at a program point: per slot, the registers its pending
 * loads write and how many messages are pending.
 */
struct wait_state {
   struct slot slots[2];
};

/*
 * AGX_WAIT_GLOBAL=0 restores the upstream per-block insertion: the first free
 * slot (else slot 0), and a wait for every slot at the end of each block.
 *
 * The default is global insertion. There is no wait at block ends: the state
 * flows along CFG edges and along layout order (union at joins, iterated to a
 * fixed point over loops), so a load is waited for only where a later
 * instruction reads or overwrites its registers, at a barrier, or on slot
 * overflow. And a load keeps filling the slot of the previous load until that
 * slot holds AGX_MAX_PENDING messages, then moves to the other slot, so a use
 * waits for its own group of loads and not for a younger group issued after
 * it (the software-pipelined pattern: issue tile k + 1, then use tile k).
 */
static bool
wait_global(void)
{
   return debug_get_bool_option("AGX_WAIT_GLOBAL", true);
}

/*
 * Assign slots and find the waits for one block from the state at its entry.
 * Updates the state to the block's exit state. Inserts the waits only when
 * emit is set, so the global analysis can run it to a fixed point first.
 *
 * XXX: Do any instructions read their sources asynchronously?
 */
static void
agx_insert_waits_block(agx_context *ctx, agx_block *block,
                       struct wait_state *st, bool global, bool emit)
{
   struct slot *slots = st->slots;
   unsigned last = slots[1].nr_pending > slots[0].nr_pending ? 1 : 0;

   /* Stores pending from an earlier block. Upstream's block-end waits ordered
    * them before any later load or atomic; keep that order (within a block,
    * as upstream, a store is not waited for before a load).
    */
   uint8_t inherited_stores = 0;
   for (unsigned slot = 0; slot < ARRAY_SIZE(st->slots); ++slot) {
      if (slots[slot].store)
         inherited_stores |= BITSET_BIT(slot);
   }

   agx_foreach_instr_in_block_safe(block, I) {
      uint8_t wait_mask = 0;

      if (instr_is_async(I) && !instr_is_store(I))
         wait_mask |= inherited_stores;

      /* Check for read-after-write */
      agx_foreach_src(I, s) {
         if (I->src[s].type != AGX_INDEX_REGISTER)
            continue;

         unsigned nr_read = agx_index_size_16(I->src[s]);
         for (unsigned slot = 0; slot < ARRAY_SIZE(st->slots); ++slot) {
            if (BITSET_TEST_COUNT(slots[slot].writes, I->src[s].value, nr_read))
               wait_mask |= BITSET_BIT(slot);
         }
      }

      /* Check for write-after-write */
      agx_foreach_dest(I, d) {
         if (I->dest[d].type != AGX_INDEX_REGISTER)
            continue;

         unsigned nr_writes = agx_index_size_16(I->dest[d]);
         for (unsigned slot = 0; slot < ARRAY_SIZE(st->slots); ++slot) {
            if (BITSET_TEST_COUNT(slots[slot].writes, I->dest[d].value,
                                  nr_writes))
               wait_mask |= BITSET_BIT(slot);
         }
      }

      /* Check for barriers */
      if (I->op == AGX_OPCODE_THREADGROUP_BARRIER ||
          I->op == AGX_OPCODE_MEMORY_BARRIER) {

         for (unsigned slot = 0; slot < ARRAY_SIZE(st->slots); ++slot) {
            if (slots[slot].nr_pending)
               wait_mask |= BITSET_BIT(slot);
         }
      }

      /* Assign a slot. The waits found above clear their slots first. */
      if (instr_is_async(I)) {
         if (global) {
            unsigned pend = (wait_mask & BITSET_BIT(last)) ? 0 : slots[last].nr_pending;
            I->scoreboard = pend < AGX_MAX_PENDING ? last : !last;
         } else {
            I->scoreboard = 0;
            for (unsigned slot = 0; slot < ARRAY_SIZE(st->slots); ++slot) {
               if (slots[slot].nr_pending == 0) {
                  I->scoreboard = slot;
                  break;
               }
            }
         }
      }

      /* Check for slot overflow */
      if (instr_is_async(I) &&
          slots[I->scoreboard].nr_pending >= AGX_MAX_PENDING)
         wait_mask |= BITSET_BIT(I->scoreboard);

      /* Insert the appropriate waits, clearing the slots */
      u_foreach_bit(slot, wait_mask) {
         if (emit) {
            agx_builder b = agx_init_builder(ctx, agx_before_instr(I));
            agx_wait(&b, slot);
         }

         BITSET_ZERO(slots[slot].writes);
         slots[slot].nr_pending = 0;
         slots[slot].store = false;
         inherited_stores &= ~BITSET_BIT(slot);
      }

      /* Record access */
      if (instr_is_async(I)) {
         agx_foreach_dest(I, d) {
            if (agx_is_null(I->dest[d]))
               continue;

            assert(I->dest[d].type == AGX_INDEX_REGISTER);
            BITSET_SET_COUNT(slots[I->scoreboard].writes, I->dest[d].value,
                             agx_index_size_16(I->dest[d]));
         }

         slots[I->scoreboard].nr_pending++;
         slots[I->scoreboard].store |= global && instr_is_store(I);
         last = I->scoreboard;
      }
   }
}

/*
 * Insert waits within a block, waiting for everything at the end of the block
 * (upstream behaviour, AGX_WAIT_GLOBAL=0).
 */
static void
agx_insert_waits_local(agx_context *ctx, agx_block *block)
{
   struct wait_state st = {0};
   agx_insert_waits_block(ctx, block, &st, false, true);

   /* If there are outstanding messages, wait for them. We don't do this for the
    * exit block, though, since nothing else will execute in the shader so
    * waiting is pointless.
    */
   if (block != agx_exit_block(ctx)) {
      agx_builder b = agx_init_builder(ctx, agx_after_block_logical(block));

      for (unsigned slot = 0; slot < ARRAY_SIZE(st.slots); ++slot) {
         if (st.slots[slot].nr_pending)
            agx_wait(&b, slot);
      }
   }
}

/* dst |= src (union of written registers, maximum of pending messages) */
static bool
wait_state_join(struct wait_state *dst, const struct wait_state *src)
{
   bool progress = false;

   for (unsigned slot = 0; slot < ARRAY_SIZE(dst->slots); ++slot) {
      for (unsigned w = 0; w < BITSET_WORDS(AGX_NUM_REGS); ++w) {
         BITSET_WORD v = dst->slots[slot].writes[w] | src->slots[slot].writes[w];
         progress |= v != dst->slots[slot].writes[w];
         dst->slots[slot].writes[w] = v;
      }

      if (src->slots[slot].nr_pending > dst->slots[slot].nr_pending) {
         dst->slots[slot].nr_pending = src->slots[slot].nr_pending;
         progress = true;
      }

      if (src->slots[slot].store && !dst->slots[slot].store) {
         dst->slots[slot].store = true;
         progress = true;
      }
   }

   return progress;
}

/*
 * Global wait insertion: the entry state of each block is the union of its
 * predecessors' exit states and the exit state of the block laid out before it;
 * a block that a skip jump may land at the end of also exits with the skip
 * source's entry state (skip_target),
 * iterated (in source order) until no entry state grows. Entry states only grow
 * and are bounded, so this terminates. Then the waits are emitted once from the
 * final entry states. No block-end waits.
 */
/* The only instruction of a block, or NULL */
static agx_instr *
single_instr(agx_block *block)
{
   if (!block || !list_is_singular(&block->instructions))
      return NULL;

   return agx_first_instr(block);
}

/*
 * The block that agx_opt_jmp_none (after this pass) may make the start of
 * `block` jump to the end of, skipping every wait in between when no lane is
 * active. NULL if none.
 *  - An else jumps to the end of the else arm's last block (its target).
 *  - A nest-1 break_if jumps to the end of the loop's last block. break_if is
 *    made later by agx_opt_break_if at the start of the block after
 *    "if (nest 1); { break (nest 2) }; { [else;] pop_exec (nest 1) }", so match
 *    that.
 */
static agx_block *
skip_target(agx_context *ctx, agx_block *block)
{
   agx_instr *first = agx_first_instr(block);
   if (first && first->target &&
       (first->op == AGX_OPCODE_ELSE_ICMP || first->op == AGX_OPCODE_ELSE_FCMP))
      return first->target;

   if (first && first->target && first->nest == 1 &&
       (first->op == AGX_OPCODE_BREAK_IF_ICMP ||
        first->op == AGX_OPCODE_BREAK_IF_FCMP))
      return agx_prev_block(first->target);

   if (block == agx_start_block(ctx))
      return NULL;

   agx_block *else_block = agx_prev_block(block);
   if (else_block == agx_start_block(ctx))
      return NULL;

   agx_block *then_block = agx_prev_block(else_block);
   if (then_block == agx_start_block(ctx))
      return NULL;

   /* The else block is a pop_exec, after an else that agx_opt_empty_else
    * deletes later.
    */
   agx_instr *pop = agx_last_instr(else_block), *brk = single_instr(then_block);
   agx_instr *else_ = agx_first_instr(else_block);
   if (else_ != pop &&
       (list_length(&else_block->instructions) != 2 ||
        (else_->op != AGX_OPCODE_ELSE_ICMP && else_->op != AGX_OPCODE_ELSE_FCMP)))
      return NULL;

   agx_instr *if_ = agx_last_instr(agx_prev_block(then_block));

   if (pop && pop->op == AGX_OPCODE_POP_EXEC && pop->nest == 1 && brk &&
       brk->op == AGX_OPCODE_BREAK && brk->nest == 2 && brk->target && if_ &&
       (if_->op == AGX_OPCODE_IF_ICMP || if_->op == AGX_OPCODE_IF_FCMP) &&
       if_->nest == 1)
      return agx_prev_block(brk->target);

   return NULL;
}

static void
agx_insert_waits_global(agx_context *ctx)
{
   struct wait_state *in = calloc(ctx->num_blocks, sizeof(*in));
   struct wait_state *out = calloc(ctx->num_blocks, sizeof(*out));
   bool progress = true;

   /* Skip sources per landing block, as linked lists: head[T], then next[S] */
   agx_block **skip_head = calloc(ctx->num_blocks, sizeof(*skip_head));
   agx_block **skip_next = calloc(ctx->num_blocks, sizeof(*skip_next));
   agx_foreach_block(ctx, S) {
      agx_block *T = skip_target(ctx, S);
      if (T) {
         skip_next[S->index] = skip_head[T->index];
         skip_head[T->index] = S;
      }
   }

   while (progress) {
      progress = false;
      agx_block *layout_prev = NULL;

      agx_foreach_block(ctx, block) {
         struct wait_state *bin = &in[block->index];

         agx_foreach_predecessor(block, pred)
            progress |= wait_state_join(bin, &out[(*pred)->index]);

         /* The hardware runs divergent if/else arms one after the other with
          * lanes masked, so a block can also run straight after the block laid
          * out before it even when the CFG has no such edge (then -> else).
          * Registers of those sibling arms may be shared by allocation.
          */
         if (layout_prev)
            progress |= wait_state_join(bin, &out[layout_prev->index]);
         layout_prev = block;

         out[block->index] = *bin;
         agx_insert_waits_block(ctx, block, &out[block->index], true, false);

         /* A skip from S lands at the end of this block with S's entry state.
          * Out-states are rebuilt each sweep, so only entry-state growth counts
          * as progress (the exits are a function of the entries).
          */
         for (agx_block *S = skip_head[block->index]; S;
              S = skip_next[S->index])
            wait_state_join(&out[block->index], &in[S->index]);
      }
   }

   agx_foreach_block(ctx, block) {
      struct wait_state st = in[block->index];
      agx_insert_waits_block(ctx, block, &st, true, true);

      /* A shader part without stop runs into the next part: leave nothing
       * pending behind it.
       */
      if (block == agx_exit_block(ctx) && ctx->key->no_stop) {
         agx_builder b = agx_init_builder(ctx, agx_after_block(block));

         for (unsigned slot = 0; slot < ARRAY_SIZE(st.slots); ++slot) {
            if (st.slots[slot].nr_pending)
               agx_wait(&b, slot);
         }
      }
   }

   free(in);
   free(out);
   free(skip_head);
   free(skip_next);
}

/*
 * Assign scoreboard slots to asynchronous instructions and insert waits for the
 * appropriate hazard tracking.
 */
void
agx_insert_waits(agx_context *ctx)
{
   if (wait_global() && !(agx_compiler_debug & AGX_DBG_WAIT)) {
      agx_insert_waits_global(ctx);
      return;
   }

   agx_foreach_block(ctx, block) {
      if (agx_compiler_debug & AGX_DBG_WAIT)
         agx_insert_waits_trivial(ctx, block);
      else
         agx_insert_waits_local(ctx, block);
   }
}
