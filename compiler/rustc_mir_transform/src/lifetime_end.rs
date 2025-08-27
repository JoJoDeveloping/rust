//! Add information about local lifetimes into the MIR, by
//! creating and inserting LifetimeEnd statements at the right places,
//! and by adding the correct entries into the local_lifetimes array.

use std::mem;

use itertools::Itertools;
use rustc_borrowck::ConstraintSccIndex;
use rustc_borrowck::consumers::{BodyWithBorrowckFacts, DetailedRegionOrigin};
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_hir::def_id::LocalDefId;
use rustc_index::IndexVec;
use rustc_middle::mir::{
    Body, CallLifetimeInstantiation, LocalLifetime, LocalLifetimeData, Location, SourceInfo,
    Statement, StatementKind, UniversalLifetimeKind, dump_mir,
};
use rustc_middle::ty::{self, Region, RegionKind, RegionVid, TyCtxt, fold_regions};
use rustc_span::DUMMY_SP;

use crate::pass_manager::MirPass;

pub(crate) struct InsertLifetimeEndInformation<'tcx>(
    pub(crate) Option<FxHashMap<LocalDefId, BodyWithBorrowckFacts<'tcx>>>,
);

#[allow(dead_code)]
#[derive(Debug)]
enum RegionEnd {
    Local(FxHashSet<Location>),
    OutlivesUniversals(Box<Vec<RegionVid>>),
}
struct LifetimeEndInsertion<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    region_data: &'a BodyWithBorrowckFacts<'tcx>,
    ending_at: FxHashMap<Location, Vec<(ConstraintSccIndex, LocalLifetime)>>,
    local_lifetimes: IndexVec<LocalLifetime, LocalLifetimeData<'tcx>>,
    cached_regions_by_extra: FxHashMap<DetailedRegionOrigin<'tcx>, RegionVid>,
    static_region_vid: RegionVid,
}

impl<'a, 'tcx> LifetimeEndInsertion<'a, 'tcx> {
    fn new(tcx: TyCtxt<'tcx>, region_data: &'a BodyWithBorrowckFacts<'tcx>) -> Self {
        let mut cached_regions_by_extra = FxHashMap::default();
        let mut static_region_vid = None;
        for (&vid, &data) in region_data.extra_info.iter() {
            let ov = cached_regions_by_extra.insert(data, vid);
            assert!(ov.is_none());
            if matches!(data, DetailedRegionOrigin::Static) {
                static_region_vid = Some(vid);
            }
        }
        Self {
            tcx,
            region_data,
            ending_at: FxHashMap::default(),
            local_lifetimes: IndexVec::new(),
            cached_regions_by_extra,
            static_region_vid: static_region_vid.unwrap(),
        }
    }

    /// Computes the first location after a region ends, for all "ends".
    fn compute_region_ends(&self, vid: RegionVid) -> (RegionEnd, ConstraintSccIndex) {
        let region_data = self.region_data;
        let sccs = region_data.region_inference_context.constraint_sccs();
        let scc = sccs.scc(vid);
        let ivals = region_data.region_inference_context.inferred_values();

        let outlived_global_regions = ivals.universal_regions_outlived_by(scc).collect::<Vec<_>>();
        if !outlived_global_regions.is_empty() {
            return (RegionEnd::OutlivesUniversals(Box::new(outlived_global_regions)), scc);
        }

        let mut res = FxHashSet::default();
        for (bb, bbdata) in region_data.body.basic_blocks.iter_enumerated() {
            for statement_index in 0..bbdata.statements.len() {
                let cur_loc = Location { block: bb, statement_index };
                let succ_loc = cur_loc.successor_within_block();
                if ivals.is_live_at(scc, cur_loc) && !ivals.is_live_at(scc, succ_loc) {
                    res.insert(succ_loc);
                }
            }
            let last_loc = Location { block: bb, statement_index: bbdata.statements.len() };
            if ivals.is_live_at(scc, last_loc) {
                for succ in bbdata.terminator().successors() {
                    let first_loc = Location { block: succ, statement_index: 0 };
                    if !ivals.is_live_at(scc, first_loc) {
                        res.insert(first_loc);
                    }
                }
            }
        }
        (RegionEnd::Local(res), scc)
    }

    fn local_lifetime_from_region(&mut self, vid: RegionVid) -> LocalLifetime {
        let (end, scc) = self.compute_region_ends(vid);
        match end {
            RegionEnd::OutlivesUniversals(region_vids) => {
                let mut global_kinds = Vec::new();
                for region_vid in region_vids.into_iter() {
                    let info = self.region_data.extra_info.get(&region_vid);
                    let global_kind = match info {
                        None | Some(DetailedRegionOrigin::LateBoundCallLifetime { .. }) => {
                            panic!(
                                "We for {vid:?} got outlives universal with rv {region_vid:?} which is {info:?} which makes no sense?!"
                            );
                        }
                        Some(DetailedRegionOrigin::ClosureEnv) => UniversalLifetimeKind::ClosureEnv,
                        Some(DetailedRegionOrigin::FreeUniversalEarlyBound { idx, .. }) => {
                            UniversalLifetimeKind::EarlyBound(*idx, None)
                        }
                        Some(DetailedRegionOrigin::FreeUniversalLateBound { idx }) => {
                            UniversalLifetimeKind::LateBound(*idx)
                        }
                        Some(DetailedRegionOrigin::Static) => {
                            return self.local_lifetimes.push(LocalLifetimeData::Static);
                        }
                        Some(DetailedRegionOrigin::FnBodyUniversal) => {
                            continue;
                        }
                        Some(DetailedRegionOrigin::CVariadics) => {
                            todo!();
                            // continue;
                        }
                        Some(DetailedRegionOrigin::RecursiveScopeUniversal { region: _region }) => {
                            todo!()
                        }
                    };
                    global_kinds.push(global_kind);
                }
                let res = self
                    .local_lifetimes
                    .push(LocalLifetimeData::PastFunctionEnd(Box::new(global_kinds)));
                res
            }
            RegionEnd::Local(hash_set) => {
                let ll = self.local_lifetimes.push(LocalLifetimeData::LocalEnd);

                // ending_at is not iterated, and each contained vector is later sorted
                // so there is no query imprecision
                #[allow(rustc::potential_query_instability)]
                for loc in hash_set.iter().copied() {
                    self.ending_at.entry(loc).or_default().push((scc, ll));
                }
                ll
            }
        }
    }

    fn find_and_process_lifetime(
        &mut self,
        which: DetailedRegionOrigin<'tcx>,
    ) -> (LocalLifetime, RegionVid) {
        let &vid = self
            .cached_regions_by_extra
            .get(&which)
            .unwrap_or_else(|| panic!("unwrap failed when getting {which:?}"));
        assert_eq!(self.region_data.extra_info.get(&vid), Some(&which));
        (self.local_lifetime_from_region(vid), vid)
    }

    fn process_call_terminators(&mut self, body: &mut Body<'tcx>) {
        for (bb, bbdata) in body.basic_blocks.as_mut_preserves_cfg().iter_enumerated_mut() {
            let loc = Location { block: bb, statement_index: bbdata.statements.len() };
            match &mut bbdata.terminator_mut().kind {
                rustc_middle::mir::TerminatorKind::Call { func, starting_lifetimes, .. }
                | rustc_middle::mir::TerminatorKind::TailCall {
                    func, starting_lifetimes, ..
                } => {
                    assert!(starting_lifetimes.is_none());
                    let starting_lifetimes =
                        starting_lifetimes.insert(Box::new(CallLifetimeInstantiation {
                            early_bound: vec![],
                            late_bound: vec![],
                        }));
                    let sig_erased = {
                        let func_ty = func.ty(&body.local_decls, self.tcx);
                        match func_ty.kind() {
                            ty::FnDef(..) | ty::FnPtr(..) => func_ty.fn_sig(self.tcx),
                            _ => unreachable!(),
                        }
                    };
                    let sig_from_renumbering = {
                        match &self.region_data.body.basic_blocks.get(bb).unwrap().terminator().kind
                        {
                            rustc_middle::mir::TerminatorKind::Call { func, .. }
                            | rustc_middle::mir::TerminatorKind::TailCall { func, .. } => {
                                let func_ty = func.ty(&self.region_data.body.local_decls, self.tcx);
                                match func_ty.kind() {
                                    ty::FnDef(..) | ty::FnPtr(..) => func_ty.fn_sig(self.tcx),
                                    _ => unreachable!(),
                                }
                            }
                            _ => unreachable!(),
                        }
                    };

                    assert_eq!(sig_from_renumbering.bound_vars(), sig_erased.bound_vars());

                    // Visit the early-bound regions. For these, we can look at the type, which was affected by region renumbering, to get the NLL region variable.
                    // This visits some regions several times, but this is unavoidable complexity since e.g. a cast could introduce such unnecessarily duplicated regions anyways.
                    let _ = fold_regions(self.tcx, sig_from_renumbering, |region, _index| {
                        let var = match region.kind() {
                            RegionKind::ReStatic => self.static_region_vid,
                            RegionKind::ReVar(v) => v,
                            _ => unreachable!(),
                        };
                        starting_lifetimes.early_bound.push(self.local_lifetime_from_region(var));
                        region
                    });

                    // Now we visit the late-bound regions, for which we must go via the detailed region origin
                    // recorded during type-checking of the call.
                    let _ = self.tcx.instantiate_bound_regions(sig_erased, |br| {
                        let (ll, vid) = self.find_and_process_lifetime(
                            DetailedRegionOrigin::LateBoundCallLifetime {
                                call: loc,
                                replaced: br.var,
                            },
                        );
                        starting_lifetimes.late_bound.push(ll);
                        Region::new_var(self.tcx, vid)
                    });
                }
                _ => {}
            }
        }
    }

    fn finish(mut self, body: &mut Body<'tcx>) {
        // We're just modifying each vector, the iteration order does not matter her.
        #[allow(rustc::potential_query_instability)]
        {
            // Sort by SCC index.
            // SCC indexes are already toposorted by construction, so that afterwards in this vector,
            // SCCs corresponding to a lifetime 'a come before those for 'b when 'b: 'a ('b outlives 'a).
            // In other words, those ending earlier come first.
            self.ending_at.iter_mut().for_each(|x| x.1.sort());
        }

        // go through the CFG, inserting statements everywhere.
        for (bb, bbdata) in body.basic_blocks_mut().iter_enumerated_mut() {
            let old_statements = mem::take(&mut bbdata.statements);
            let old_terminator_idx = old_statements.len();
            let mut at_pos_action =
                |statement_index, source_info, pusher: &mut dyn FnMut(Statement<'tcx>)| {
                    let here = Location { block: bb, statement_index };
                    let ending_here = self.ending_at.remove(&here);
                    let grouped =
                        ending_here.into_iter().flat_map(|x| x.into_iter()).group_by(|x| x.0);
                    for (_scc, things) in grouped.into_iter() {
                        pusher(Statement::new(
                            source_info,
                            StatementKind::LocalLifetimeEnd(Box::new(
                                things.map(|x| x.1).collect::<Vec<_>>(),
                            )),
                        ));
                    }
                };
            for (statement_index, stmt) in old_statements.into_iter().enumerate() {
                at_pos_action(statement_index, stmt.source_info, &mut |x| {
                    bbdata.statements.push(x)
                });
                bbdata.statements.push(stmt);
            }
            at_pos_action(
                old_terminator_idx,
                bbdata
                    .terminator
                    .as_ref()
                    .map(|x| x.source_info)
                    .unwrap_or_else(|| SourceInfo::outermost(DUMMY_SP)),
                &mut |x| bbdata.statements.push(x),
            );
            // terminator remains in place
        }

        body.local_lifetimes = self.local_lifetimes;
    }
}

fn is_trivial_body<'tcx>(body: &Body<'tcx>) -> bool {
    body.basic_blocks.len() <= 1
        && body.basic_blocks.iter().all(|x| {
            x.statements.is_empty()
                && matches!(&x.terminator().kind, rustc_middle::mir::TerminatorKind::Unreachable)
        })
}

impl<'tcx> MirPass<'tcx> for InsertLifetimeEndInformation<'tcx> {
    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        let Some(x) = self.0.as_ref() else {
            // println!("Borrowck information for {:?} seems to be missing?", body.source.def_id());
            return;
        };
        // println!("Running my custom pass on {:?} wohoo! ({:?})", body.source.def_id(), body.span);
        assert!(x.contains_key(&body.source.def_id().as_local().unwrap()));
        assert!(body.local_lifetimes.is_empty());

        if is_trivial_body(body) {
            return;
        }

        let region_data = x.get(&body.source.def_id().expect_local()).unwrap();
        assert_eq!(body.basic_blocks.len(), region_data.body.basic_blocks.len());
        assert_eq!(body.local_decls.len(), region_data.body.local_decls.len());

        let mut handler = LifetimeEndInsertion::new(tcx, region_data);

        handler.process_call_terminators(body);

        handler.finish(body);
        dump_mir(tcx, false, "lifetime_end", &0, body, |_, _| Ok(()));
    }

    fn is_enabled(&self, sess: &rustc_session::Session) -> bool {
        sess.opts.unstable_opts.mir_emit_lifetime_information
    }

    fn is_required(&self) -> bool {
        true
    }
}
